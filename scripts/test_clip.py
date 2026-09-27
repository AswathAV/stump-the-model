import os
import torch

from PIL import Image
from transformers import CLIPProcessor, CLIPModel


MODEL_NAME = "openai/clip-vit-base-patch32"

IMAGE_PATH = "data/catalogue/images/shoe_00001.jpg"


print("Checking image...")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

print("Image exists:", True)


# -------------------------
# Load image
# -------------------------

image = Image.open(IMAGE_PATH).convert("RGB")

print("Image size:", image.size)


# -------------------------
# Load CLIP
# -------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

print("Loading CLIP...")

model = CLIPModel.from_pretrained(
    MODEL_NAME
)

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("CLIP loaded successfully")


# -------------------------
# Process image
# -------------------------

inputs = processor(
    images=image,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# -------------------------
# Generate CLIP embedding
# -------------------------

print("Generating embedding...")

with torch.no_grad():

    vision_output = model.vision_model(
        **inputs
    )

    pooled_output = vision_output.pooler_output

    image_features = model.visual_projection(
        pooled_output
    )


# -------------------------
# Normalize
# -------------------------

image_features = image_features / image_features.norm(
    dim=-1,
    keepdim=True
)


# -------------------------
# Print result
# -------------------------

print("\nSUCCESS!")

print(
    "Embedding type:",
    type(image_features)
)

print(
    "Embedding shape:",
    image_features.shape
)

print(
    "First 10 values:"
)

print(
    image_features[0][:10]
)