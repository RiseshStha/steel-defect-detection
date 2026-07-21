import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.datasets.classification_dataset import ClassificationDataset
from src.datasets.segmentation_dataset import SteelSegmentationDataset
from src.datasets.transforms import get_val_classification_transform
from src.models.cbam_resnet18 import CBAMResNet18
from src.models.resnet18_classifier import ResNet18Classifier
from src.models.unet_cbam import UNetCBAM
from src.models.unet_resnet18 import UNetResNet18


@dataclass(frozen=True)
class RuntimeResult:
    task: str
    model: str
    parameters: int
    trainable_parameters: int
    checkpoint_size_mb: float
    epochs_recorded: int
    total_training_time_min: str
    mean_epoch_time_sec: str
    inference_time_ms_per_image: float
    throughput_images_per_sec: float
    peak_inference_gpu_memory_mb: str
    device: str
    notes: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect runtime, parameter, checkpoint size, and inference benchmarks."
    )

    parser.add_argument(
        "--classification-val-csv",
        default="data/processed/val_split.csv",
        help="Classification validation split CSV.",
    )

    parser.add_argument(
        "--segmentation-train-csv",
        default="data/train.csv",
        help="Segmentation train CSV used to reconstruct the validation split.",
    )

    parser.add_argument(
        "--classification-image-dir",
        default="data/train_images",
        help="Directory containing classification images.",
    )

    parser.add_argument(
        "--segmentation-image-dir",
        default="data/train_images",
        help="Directory containing segmentation images.",
    )

    parser.add_argument(
        "--baseline-classification-log",
        default="logs/baseline_training.csv",
        help="Baseline ResNet18 training log.",
    )

    parser.add_argument(
        "--cbam-classification-log",
        default="logs/cbam_training.csv",
        help="CBAM-ResNet18 training log.",
    )

    parser.add_argument(
        "--baseline-segmentation-log",
        default="logs/baseline_history.csv",
        help="Baseline U-Net training history.",
    )

    parser.add_argument(
        "--cbam-segmentation-log",
        default="logs/cbam_history.csv",
        help="CBAM U-Net training history.",
    )

    parser.add_argument(
        "--baseline-classification-checkpoint",
        default="checkpoints/baseline_resnet18_best.pth",
        help="Baseline ResNet18 checkpoint.",
    )

    parser.add_argument(
        "--cbam-classification-checkpoint",
        default="checkpoints/cbam_best.pth",
        help="CBAM-ResNet18 checkpoint.",
    )

    parser.add_argument(
        "--baseline-segmentation-checkpoint",
        default="checkpoints/baseline_unet_best.pth",
        help="Baseline U-Net checkpoint.",
    )

    parser.add_argument(
        "--cbam-segmentation-checkpoint",
        default="checkpoints/cbam_unet_best.pth",
        help="CBAM U-Net checkpoint.",
    )

    parser.add_argument(
        "--classification-batch-size",
        type=int,
        default=32,
        help="Batch size for classification inference benchmarking.",
    )

    parser.add_argument(
        "--segmentation-batch-size",
        type=int,
        default=8,
        help="Batch size for segmentation inference benchmarking.",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=20,
        help="Maximum timed batches per benchmark.",
    )

    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=3,
        help="Warmup batches before timing.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Zero is safest for Windows benchmarking.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Segmentation image size used during training.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Segmentation validation split seed.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/runtime",
        help="Directory where runtime analysis files will be written.",
    )

    return parser.parse_args()


def parameter_count(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def checkpoint_size_mb(checkpoint_path):
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return path.stat().st_size / (1024 * 1024)


def load_state_dict(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    return model


def build_classification_loader(args):
    val_df = pd.read_csv(args.classification_val_csv)
    dataset = ClassificationDataset(
        val_df,
        args.classification_image_dir,
        transform=get_val_classification_transform(),
    )
    return DataLoader(
        dataset,
        batch_size=args.classification_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_segmentation_loader(args):
    train_df = pd.read_csv(args.segmentation_train_csv)
    _, val_df = train_test_split(
        train_df,
        test_size=0.2,
        shuffle=True,
        random_state=args.seed,
    )
    dataset = SteelSegmentationDataset(
        dataframe=val_df,
        image_dir=args.segmentation_image_dir,
        transform=None,
        image_size=(args.image_size, args.image_size),
    )
    return DataLoader(
        dataset,
        batch_size=args.segmentation_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def benchmark_inference(model, loader, device, warmup_batches, max_batches):
    model.eval()

    iterator = iter(loader)

    for _ in range(warmup_batches):
        try:
            images, _ = next(iterator)
        except StopIteration:
            break

        images = images.to(device, non_blocking=True)
        _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    timed_images = 0
    timed_batches = 0
    start_time = time.perf_counter()

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        _ = model(images)
        timed_images += images.size(0)
        timed_batches += 1

        if timed_batches >= max_batches:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    if timed_images == 0:
        raise ValueError("No images were available for inference benchmarking.")

    milliseconds_per_image = (elapsed / timed_images) * 1000.0
    throughput = timed_images / elapsed

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        peak_memory_text = f"{peak_memory:.2f}"
    else:
        peak_memory_text = "not_available_cpu_run"

    return milliseconds_per_image, throughput, peak_memory_text


def summarize_training_log(log_path):
    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    df = pd.read_csv(log_path)

    if df.empty:
        raise ValueError(f"Training log is empty: {log_path}")

    epochs = len(df)

    if "epoch_time" not in df.columns:
        return epochs, "not_recorded", "not_recorded"

    total_seconds = float(df["epoch_time"].sum())
    mean_seconds = float(df["epoch_time"].mean())
    return epochs, f"{total_seconds / 60.0:.4f}", f"{mean_seconds:.4f}"


def build_model(task, model_name):
    if task == "classification" and model_name == "baseline":
        return ResNet18Classifier(pretrained=False, num_classes=4)

    if task == "classification" and model_name == "cbam":
        return CBAMResNet18(pretrained=False, num_classes=4)

    if task == "segmentation" and model_name == "baseline":
        return UNetResNet18(num_classes=4, pretrained=False)

    if task == "segmentation" and model_name == "cbam":
        return UNetCBAM(num_classes=4)

    raise ValueError(f"Unsupported model: task={task}, model={model_name}")


def collect_result(
    task,
    model_label,
    model_name,
    checkpoint_path,
    log_path,
    loader,
    device,
    args,
):
    model = build_model(task, model_name)
    model = load_state_dict(model, checkpoint_path, device)
    model = model.to(device)

    total_parameters, trainable_parameters = parameter_count(model)
    size_mb = checkpoint_size_mb(checkpoint_path)
    epochs, total_training_time_min, mean_epoch_time_sec = summarize_training_log(log_path)
    inference_ms, throughput, peak_gpu_memory = benchmark_inference(
        model,
        loader,
        device,
        args.warmup_batches,
        args.max_batches,
    )

    notes = ""

    if total_training_time_min == "not_recorded":
        notes = "Epoch time was not saved in the segmentation history CSV."

    return RuntimeResult(
        task=task,
        model=model_label,
        parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        checkpoint_size_mb=size_mb,
        epochs_recorded=epochs,
        total_training_time_min=total_training_time_min,
        mean_epoch_time_sec=mean_epoch_time_sec,
        inference_time_ms_per_image=inference_ms,
        throughput_images_per_sec=throughput,
        peak_inference_gpu_memory_mb=peak_gpu_memory,
        device=str(device),
        notes=notes,
    )


def write_csv(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task",
                "model",
                "parameters",
                "trainable_parameters",
                "checkpoint_size_mb",
                "epochs_recorded",
                "total_training_time_min",
                "mean_epoch_time_sec",
                "inference_time_ms_per_image",
                "throughput_images_per_sec",
                "peak_inference_gpu_memory_mb",
                "device",
                "notes",
            ],
        )
        writer.writeheader()

        for result in results:
            row = result.__dict__.copy()
            row["checkpoint_size_mb"] = f"{result.checkpoint_size_mb:.4f}"
            row["inference_time_ms_per_image"] = (
                f"{result.inference_time_ms_per_image:.4f}"
            )
            row["throughput_images_per_sec"] = (
                f"{result.throughput_images_per_sec:.4f}"
            )
            writer.writerow(row)


def markdown_value(value):
    return value if isinstance(value, str) else f"{value}"


def write_markdown(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Runtime Analysis",
        "",
        "Inference timing was benchmarked on the active PyTorch device using validation data.",
        "Segmentation training time is reported as not recorded because the existing segmentation history CSVs do not contain an epoch_time column.",
        "",
        "| Task | Model | Parameters | Checkpoint MB | Epochs | Training Time (min) | Mean Epoch Time (sec) | Inference (ms/image) | Throughput (images/sec) | Peak GPU Memory MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for result in results:
        lines.append(
            "| "
            f"{result.task} | "
            f"{result.model} | "
            f"{result.parameters:,} | "
            f"{result.checkpoint_size_mb:.2f} | "
            f"{result.epochs_recorded} | "
            f"{result.total_training_time_min} | "
            f"{result.mean_epoch_time_sec} | "
            f"{result.inference_time_ms_per_image:.2f} | "
            f"{result.throughput_images_per_sec:.2f} | "
            f"{result.peak_inference_gpu_memory_mb} |"
        )

    notes = sorted({result.notes for result in results if result.notes})

    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)

    output_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classification_loader = build_classification_loader(args)
    segmentation_loader = build_segmentation_loader(args)

    results = [
        collect_result(
            "classification",
            "Baseline ResNet18",
            "baseline",
            args.baseline_classification_checkpoint,
            args.baseline_classification_log,
            classification_loader,
            device,
            args,
        ),
        collect_result(
            "classification",
            "CBAM-ResNet18",
            "cbam",
            args.cbam_classification_checkpoint,
            args.cbam_classification_log,
            classification_loader,
            device,
            args,
        ),
        collect_result(
            "segmentation",
            "Baseline U-Net",
            "baseline",
            args.baseline_segmentation_checkpoint,
            args.baseline_segmentation_log,
            segmentation_loader,
            device,
            args,
        ),
        collect_result(
            "segmentation",
            "CBAM U-Net",
            "cbam",
            args.cbam_segmentation_checkpoint,
            args.cbam_segmentation_log,
            segmentation_loader,
            device,
            args,
        ),
    ]

    csv_path = output_dir / "runtime_analysis.csv"
    markdown_path = output_dir / "runtime_analysis.md"

    write_csv(results, csv_path)
    write_markdown(results, markdown_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
