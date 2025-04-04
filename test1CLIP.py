import os
import glob
import faiss
import numpy as np
from PIL import Image
import torch
import clip
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import gradio as gr

# Define a custom dataset that loads images from a directory
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path  # Return image tensor and its path

# Set paths
base_dir = "coco128"
image_dir = os.path.join(base_dir, 'images', 'train2017')

# Image transform for dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset and loader
dataset = CustomImageDataset(image_dir=image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Extract features
image_embeddings = []
image_paths = []

for i, (image, img_path) in enumerate(dataloader):
    with torch.no_grad():
        image = image.to(device)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_embeddings.append(image_features.cpu().numpy())
        image_paths.append(img_path[0])  # store full image path

image_embeddings = np.vstack(image_embeddings)

# Create FAISS index
dimension = image_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(image_embeddings)

# Search function
def search_similar_images(query, k=5):
    if isinstance(query, Image.Image):  # Image input
        image_tensor = preprocess(query).unsqueeze(0).to(device)
        with torch.no_grad():
            query_embedding = model.encode_image(image_tensor)
    else:  # Text input
        text_tokens = clip.tokenize([query]).to(device)
        with torch.no_grad():
            query_embedding = model.encode_text(text_tokens)

    query_embedding /= query_embedding.norm(dim=-1, keepdim=True)
    query_embedding = query_embedding.cpu().numpy()
    D, I = index.search(query_embedding, k)
    return I[0]


# Gradio function
def search_and_display(query_image):
    indices = search_similar_images(query_image)
    result_images = []
    for idx in indices:
        result_images.append(Image.open(image_paths[idx]))
    return [query_image] + result_images

def image_text_search(query_input):
    # query_input could be either an image or a string
    indices = search_similar_images(query_input)
    result_images = [Image.open(image_paths[idx]) for idx in indices]
    if isinstance(query_input, Image.Image):
        return [query_input] + result_images
    else:
        return result_images  # No need to include text input in output


# Launch Gradio
def unified_search(text_query, image_query):
    if image_query is not None:
        return search_and_display(image_query)
    elif text_query.strip() != "":
        indices = search_similar_images(text_query)
        return [Image.open(image_paths[idx]) for idx in indices]
    else:
        return []

gr.Interface(
    fn=unified_search,
    inputs=[
        gr.Textbox(label="Text Query (e.g., 'pizza')", placeholder="Leave empty if uploading an image"),
        gr.Image(type="pil", label="Or upload an image")
    ],
    outputs=[gr.Image(label=f"Result {i+1}") for i in range(5)],
    title="CLIP Image & Text Search",
    description="Search for similar images using text (like 'dog', 'pizza') or upload an image."
).launch()

