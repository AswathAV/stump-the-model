import numpy as np
import faiss
import os


EMBEDDINGS_FILE = "models/catalogue_embeddings.npy"
INDEX_FILE = "index/catalogue.faiss"


os.makedirs("index", exist_ok=True)


# Load embeddings
embeddings = np.load(EMBEDDINGS_FILE)

print("Embedding shape:", embeddings.shape)


# Number of dimensions
dimension = embeddings.shape[1]


# Create FAISS index
index = faiss.IndexFlatIP(dimension)


# Add embeddings
index.add(embeddings)


print("Number of vectors in index:", index.ntotal)


# Save
faiss.write_index(
    index,
    INDEX_FILE
)


print("\nFAISS index created successfully.")

print(
    "Saved to:",
    INDEX_FILE
)