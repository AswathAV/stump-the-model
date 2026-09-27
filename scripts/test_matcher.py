import os
import sys

# Allow importing from app/
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from PIL import Image
from app.matcher import search


# Test image
IMAGE_PATH = "data/catalogue/images/shoe_00001.jpg"


print("Loading test image...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")


print("Searching catalogue...")

results = search(
    image,
    top_k=5
)


print("\n==============================")
print("TOP 5 MATCHES")
print("==============================")


for rank, result in enumerate(
    results,
    start=1
):

    print(
        f"{rank}. "
        f"Product ID: {result['product_id']} "
        f"| Similarity: {result['similarity']:.4f}"
    )