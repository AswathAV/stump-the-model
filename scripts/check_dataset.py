import os
import pandas as pd

IMAGE_DIR = "data/catalogue/images"
METADATA_FILE = "data/catalogue/metadata.csv"

# Check metadata
df = pd.read_csv(METADATA_FILE)

print("Number of products:", len(df))
print("\nFirst 5 products:")
print(df.head())

# Check images
images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("\nNumber of images:", len(images))

# Check missing images
missing = []

for path in df["image_path"]:
    full_path = os.path.join("data/catalogue", path)

    if not os.path.exists(full_path):
        missing.append(full_path)

print("Missing images:", len(missing))

if len(df) >= 5000 and len(images) >= 5000 and len(missing) == 0:
    print("\nDATASET CHECK PASSED")
else:
    print("\nDATASET CHECK FAILED")