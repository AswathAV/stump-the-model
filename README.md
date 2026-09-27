# 👟 Stump the Model

A computer vision matching system that identifies the most similar
footwear item from a catalogue of 5,000 images.

## 🚀 Live Demo

**Deployed Application:**  
https://stump-the-model-6nd2zkh5wlcbbyj48qxq7w.streamlit.app/

---

## 📌 Overview

Stump the Model is a visual product-matching system designed to identify
a catalogue item from an input photograph.

The system uses CLIP to convert images into visual embeddings and FAISS
to perform efficient similarity search across a catalogue of 5,000
footwear images.

For each uploaded image, the system returns the top 5 most similar
catalogue items along with their similarity scores.

---

## ✨ Features

- 5,000-image footwear catalogue
- CLIP-based image embeddings
- FAISS vector similarity search
- Top-5 matching results
- Similarity scores
- Streamlit web interface
- Real-world phone-image evaluation
- Testing under difficult visual conditions

---

## 🏗️ System Architecture

```text
Input Image
     │
     ▼
CLIP Image Encoder
     │
     ▼
512-dimensional Image Embedding
     │
     ▼
FAISS Similarity Search
     │
     ▼
Top 5 Catalogue Matches
     │
     ▼
Product ID + Similarity Score