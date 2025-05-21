from matplotlib import pyplot as plt
import os
from PIL import Image
import warnings
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from langchain_openai import ChatOpenAI
import base64
from chromadb.utils.data_loaders import ImageLoader
import pytesseract

warnings.filterwarnings("ignore")
image_loader = ImageLoader()

path = "/pic/"

# 2. Setup ChromaDB with OpenCLIP
chroma_client = chromadb.PersistentClient(path="/pics.db")
embedding_function = OpenCLIPEmbeddingFunction()

vectorstore = chroma_client.get_or_create_collection(
    "vectorstore",
    embedding_function=embedding_function,
    data_loader=image_loader
)

# Clear the database if it exists, to avoid duplicate ent
vectorstore = chroma_client.get_or_create_collection(
    "vectorstore",
    embedding_function=embedding_function,
    data_loader=image_loader
)

# Add Image Embeddings with Captions
dataset_folder = "/pic/"

image_files = [f for f in os.listdir(dataset_folder) if f.endswith(("robotarm.jpg"))]

image_ids = [f"img-{i}" for i in range(len(image_files))]
image_uris = [os.path.join(dataset_folder, f) for f in image_files]

image_metadatas = []
for uri in image_uris:
    try:
        img = Image.open(uri)
        extracted_caption = pytesseract.image_to_string(img).strip()
        print(extracted_caption)
        image_metadatas.append({"caption": extracted_caption})
    except Exception as e:
        print(f"Error processing image {uri}: {e}")
        image_metadatas.append({"caption": "No caption extracted."})