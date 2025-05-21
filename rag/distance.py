import numpy as np
from PIL import Image
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
# Load and process image to RGB and resize to (224, 224)
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    return np.array(img)
while True:
    # Get embeddings
    img1 = preprocess_image(input("input 1"))
    img2 = preprocess_image(input("input 2"))
    embedding_function = OpenCLIPEmbeddingFunction()

    emb1 = embedding_function([img1])[0]
    emb2 = embedding_function([img2])[0]

    # Compute cosine distance
    def cosine_distance(a, b):
        a = np.array(a)
        b = np.array(b)
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    distance = cosine_distance(emb1, emb2)
    print(f"Cosine Distance between the two images: {distance:.4f}")
