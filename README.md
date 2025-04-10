# 🔍 CLIP Image & Text Search with FAISS

A simple and efficient image search engine using OpenAI's CLIP model and FAISS. Supports both **image-to-image** and **text-to-image** similarity search on the COCO128 dataset with a clean Gradio interface.

---

## 🚀 Features

- 🔤 Text-to-image search (e.g., search "pizza" to find related images)
- 🖼️ Image-to-image search (upload an image to find similar ones)
- ⚡ Fast similarity search using FAISS
- 🧠 CLIP (ViT-B/32) for joint image-text embeddings
- 🖥️ Gradio UI for a user-friendly experience

---

## 📁 Directory Structure

```bash
project_root/
├── test1CLIP.py
├── README.md
└── coco128/
    ├── images/
    │   └── train2017/
    │       ├── 000000000001.jpg
    │       ├── 000000000002.jpg
    │       └── ...
    └── labels/
        └── train2017/
            ├── 000000000001.txt
            ├── 000000000002.txt
            └── ...
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sanzy874/image-search-coco128.git
cd image-search-coco128
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision faiss-cpu gradio ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

💡 You can use faiss-gpu instead of faiss-cpu for GPU acceleration.

---

## ▶️ Run the App

```bash
python3 test1CLIP.py
```
After launching, open the Gradio link shown in your terminal.

---

## 🧠 How It Works
-Loads all images from the dataset and encodes them using CLIP.

-Normalized embeddings are stored in a FAISS index.

-User provides a text or image query.

-The query is embedded with CLIP and matched against the index.

-Top-k similar images are retrieved and displayed.

---

## 🙌 Acknowledgments
-OpenAI CLIP

-FAISS

-Gradio

-COCO Dataset
