import os
import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


# =========================
# CONFIG
# =========================

MODEL_NAME = "openai/clip-vit-base-patch32"

METADATA_FILE = "data/catalogue/metadata.csv"

CATALOGUE_DIR = "data/catalogue"

OUTPUT_DIR = "models"


# =========================
# SETUP
# =========================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(
    MODEL_NAME
)

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

model.to(device)

model.eval()

print("CLIP model loaded.")


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    METADATA_FILE
)

print(
    "Number of catalogue items:",
    len(df)
)


# =========================
# EMBEDDINGS
# =========================

embeddings = []

product_ids = []

failed = []


# =========================
# PROCESS IMAGES
# =========================

with torch.no_grad():

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Generating embeddings"
    ):

        image_path = os.path.join(
            CATALOGUE_DIR,
            row["image_path"]
        )

        try:

            # Load image
            image = Image.open(
                image_path
            ).convert("RGB")


            # Process image
            inputs = processor(
                images=image,
                return_tensors="pt"
            )


            # Move tensors to CPU/GPU
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }


            # -------------------------
            # CLIP image encoder
            # -------------------------

            vision_output = model.vision_model(
                **inputs
            )


            # Get pooled visual representation
            pooled_output = (
                vision_output.pooler_output
            )


            # Project to CLIP embedding space
            image_features = (
                model.visual_projection(
                    pooled_output
                )
            )


            # Normalize
            image_features = (
                image_features
                / image_features.norm(
                    dim=-1,
                    keepdim=True
                )
            )


            # Convert to NumPy
            embedding = (
                image_features
                .cpu()
                .numpy()[0]
                .astype("float32")
            )


            # Store
            embeddings.append(
                embedding
            )

            product_ids.append(
                row["product_id"]
            )


        except Exception as e:

            print(
                f"\nERROR processing:"
                f" {image_path}"
            )

            print(
                "Reason:",
                e
            )

            failed.append(
                image_path
            )


# =========================
# CONVERT TO NUMPY
# =========================

embeddings = np.array(
    embeddings,
    dtype="float32"
)

product_ids = np.array(
    product_ids
)


# =========================
# SAVE
# =========================

np.save(
    "models/catalogue_embeddings.npy",
    embeddings
)

np.save(
    "models/product_ids.npy",
    product_ids
)


# =========================
# RESULTS
# =========================

print("\n==========================")
print("EMBEDDING GENERATION DONE")
print("==========================")

print(
    "Successful:",
    len(embeddings)
)

print(
    "Failed:",
    len(failed)
)

print(
    "Embedding shape:",
    embeddings.shape
)

print(
    "Saved:",
    "models/catalogue_embeddings.npy"
)

print(
    "Saved:",
    "models/product_ids.npy"
)

if failed:

    print("\nFailed images:")

    for path in failed[:10]:
        print(path)