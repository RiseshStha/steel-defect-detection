import subprocess

SEEDS = [42, 123, 999]
MODELS = ["baseline", "cbam"]

for model in MODELS:

    print("=" * 70)
    print(f"Running {model.upper()} experiments")
    print("=" * 70)

    for seed in SEEDS:

        print(f"\nTraining {model} | Seed {seed}\n")

        subprocess.run(
            [
                "python",
                "train_classifier.py",
                "--model",
                model,
                "--seed",
                str(seed),
            ],
            check=True,
        )

print("\nAll experiments completed successfully.")