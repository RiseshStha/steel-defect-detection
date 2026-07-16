import numpy as np
import matplotlib.pyplot as plt

from utils1 import rle_encode, rle_decode


# -------------------------
# Create simple test mask
# -------------------------

mask = np.zeros((10, 10), dtype=np.uint8)

mask[2:6, 3:8] = 1

print("Original mask:\n")
print(mask)

# -------------------------
# Encode
# -------------------------

rle = rle_encode(mask)

print("\nEncoded RLE:\n")
print(rle)

# -------------------------
# Decode
# -------------------------

decoded = rle_decode(rle, shape=(10, 10))

print("\nDecoded mask:\n")
print(decoded)

# -------------------------
# Check
# -------------------------

print("\nMasks identical?")

print(np.array_equal(mask, decoded))

# -------------------------
# Visualization
# -------------------------

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(mask, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(decoded, cmap='gray')
plt.title("Decoded")

plt.tight_layout()
plt.show()