import numpy as np
import torch
import faiss

from transformers import CLIPProcessor, CLIPModel


MODEL_NAME = "openai/clip-vit-base-patch32"

INDEX_FILE = "index/catalogue.faiss"

PRODUCT_IDS_FILE = "models/product_ids.npy"


# =========================
# DEVICE
# =========================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================
# LOAD CLIP
# =========================

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(
    MODEL_NAME
)

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

model.to(device)

model.eval()


# =========================
# LOAD FAISS
# =========================

print("Loading FAISS index...")

index = faiss.read_index(
    INDEX_FILE
)


# =========================
# LOAD PRODUCT IDS
# =========================

product_ids = np.load(
    PRODUCT_IDS_FILE,
    allow_pickle=True
)


print(
    "Catalogue vectors:",
    index.ntotal
)


# =========================
# EMBEDDING FUNCTION
# =========================

def get_embedding(image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    with torch.no_grad():

        vision_output = model.vision_model(
            **inputs
        )

        pooled_output = (
            vision_output.pooler_output
        )

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


    return (
        image_features
        .cpu()
        .numpy()
        .astype("float32")
    )


# =========================
# SEARCH
# =========================

def search(
    image,
    top_k=5
):

    query_embedding = get_embedding(
        image
    )


    scores, indices = index.search(
        query_embedding,
        top_k
    )


    results = []


    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        results.append({

            "product_id":
                str(product_ids[idx]),

            "similarity":
                float(score)

        })


    return results