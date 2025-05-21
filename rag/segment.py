from matplotlib import pyplot as plt
import os
from PIL import Image
import warnings
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig
import torch
import segment_anything

warnings.filterwarnings("ignore")

# --- Load Models and Embedding Function (as before) ---
model_id = "allenai/MolmoE-1B-0924"
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=False
)
def load_model(model_id, quant_config: BitsAndBytesConfig = None, dtype="auto", device="cuda"):
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype="auto", device_map=device
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
        quantization_config=quant_config,
    )
    return model, processor
    
model, processor = load_model(model_id, quant_config=quant_config, dtype="auto", device="cuda")
embedding_function = OpenCLIPEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="/pct.db")
vectorstore = chroma_client.get_collection("vectorstore", embedding_function=embedding_function)

# --- Load Segment Anything Model (as before) ---
sam_checkpoint = "sam2/sam2.1_hiera_small.pt"
sam = segment_anything.sam_model_registry["small"](checkpoint=sam_checkpoint)
sam.to(device="cuda")
mask_generator = segment_anything.SamAutomaticMaskGenerator(sam)

def segment_everything(image_path):
    try:
        image = np.array(Image.open(image_path).convert("RGB"))
        masks = mask_generator.generate(image)
        return masks
    except Exception as e:
        print(f"Error segmenting image: {e}")
        return []

def embed_region_from_mask(image_path, mask):
    try:
        image = Image.open(image_path).convert("RGB")
        masked_image = Image.composite(image, Image.new('RGB', image.size, (0,0,0)), Image.fromarray(mask.astype(np.uint8) * 255))
        masked_image = masked_image.resize((224, 224))
        masked_array = np.array(masked_image)
        embedding = embedding_function([masked_array])[0]
        return embedding
    except Exception as e:
        print(f"Error embedding masked region: {e}")
        return None

def rag_image_query_segment_everything():
    image_path = input("Enter the path to your query image: \n")
    user_query = input("Enter your query (e.g., 'point to dog'): \n")

    # 1. Segment everything
    segments = segment_everything(image_path)

    if not segments:
        print("No objects segmented in the image.")
        return

    best_match = None
    max_similarity = -1

    # 2. Embed and search for each segment
    for segment in segments:
        mask = segment['segmentation']
        bbox = segment['bbox'] # You might need this for pointing later
        segment_embedding = embed_region_from_mask(image_path, mask)

        if segment_embedding is not None:
            # 3. Search for similar objects in the database
            similar_objects = vectorstore.query(
                query_embeddings=[segment_embedding],
                n_results=1,  # Get the top match for each segment
                include=["uris", "distances", "metadatas"]
            )

            if similar_objects and similar_objects.get("metadatas") and similar_objects["metadatas"][0]:
                match_metadata = similar_objects["metadatas"][0][0]
                similarity_score = similar_objects["distances"][0][0]

                # 4. Determine the "best" match (e.g., highest similarity)
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_match = {"mask": mask, "bbox": bbox, "metadata": match_metadata}

    if best_match:
        print(f"Best match found (label in database): {best_match['metadata'].get('label')}") # Assuming your database metadata has a 'label'
        # 5. Point to the segmented object in the original image using Molmo
        if "point" in user_query.lower():
            response = custom_invoke_with_pointing(
                model=model,
                processor=processor,
                image_path=image_path,
                user_query=user_query,
                mask=best_match['mask'] # Pass the mask of the best matching segment
            )
            print(f"Molmo Pointing Response: {response}")
    else:
        print("No similar objects found in the database for any segment.")

def main():
    rag_image_query_segment_everything()

if __name__ == "__main__":
    main()