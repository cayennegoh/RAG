import warnings
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from langchain_openai import ChatOpenAI
import os
from PIL import Image
import pytesseract
import torch
import base64
from chromadb.utils.data_loaders import ImageLoader
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig
import cv2  # For drawing bounding boxes
import re
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # Suppress warnings globally

# --- Configuration ---
dataset_folder = "/pic/dataset"
compare_folder = "/pic/compare"
dataset_db_path = "/pictures.db"  # Original dataset DB
compare_db_path = "/pictures_compare.db"  # Separate DB for cleaned images
model_id = "allenai/MolmoE-1B-0924"
embedding_function = OpenCLIPEmbeddingFunction()

# --- Load Molmo Model and Processor ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=False
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

# --- Function to Remove Text from Image ---
def remove_text_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    mask = np.zeros_like(gray)
    for i in range(len(data["text"])):
        if int(data["conf"][i]) > 50:
            (x, y, w, h) = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))

# --- Initialize ChromaDB for Cleaned Images ---
chroma_client_compare = chromadb.PersistentClient(path=compare_db_path)
vectorstore_compare = chroma_client_compare.get_or_create_collection(
    "cleaned_images",
    embedding_function=embedding_function,
    data_loader=ImageLoader()
)

# --- Initialize ChromaDB for Original Images (with captions) ---
chroma_client_original = chromadb.PersistentClient(path=dataset_db_path)
vectorstore_original = chroma_client_original.get_or_create_collection(
    "vectorstore",
    embedding_function=embedding_function,
    data_loader=ImageLoader()
)

# --- Function to Index Original Images (with text and image embeddings) ---
def index_original_images(image_files, image_ids, image_uris):
    image_metadatas = []
    text_embeddings = []
    image_embeddings = []

    for uri in image_uris:
        try:
            img = Image.open(uri)
            extracted_caption = pytesseract.image_to_string(img).strip()
            print(f"Extracted caption for {uri}: {extracted_caption}")
            # Generate text embeddings
            text_embedding = embedding_function([extracted_caption])
            text_embeddings.append(text_embedding)
            # Generate image embeddings (passing the image itself)
            img = img.convert('RGB')
            img = img.resize((224, 224))  # Resizing image to 224x224 for OpenCLIP model
            img_array = np.array(img)
            image_embedding = embedding_function([img_array])
            image_embeddings.append(image_embedding)
            image_metadatas.append({"caption": extracted_caption, "uri": uri})
        except Exception as e:
            print(f"Error processing image {uri}: {e}")
            text_embeddings.append(None)
            image_embeddings.append(None)
            image_metadatas.append({"caption": "No caption extracted.", "uri": uri})

    # Store both text and image embeddings in the vectorstore
    for i, image_id in enumerate(image_ids):
        if text_embeddings[i] is not None and image_embeddings[i] is not None:
            vectorstore_original.add(
                ids=[image_id + "-text"],
                embeddings=text_embeddings[i],
                metadatas=[image_metadatas[i]],
                uris=[image_uris[i]]
            )
            vectorstore_original.add(
                ids=[image_id + "-image"],
                embeddings=image_embeddings[i],
                metadatas=[image_metadatas[i]],
                uris=[image_uris[i]]
            )
    print(f"Text and Image Embeddings added to the original database. Total items: {vectorstore_original.count()}")

# --- Function to Index Cleaned Images ---
def index_cleaned_images(image_uris):
    os.makedirs(compare_folder, exist_ok=True)
    cleaned_files = [f for f in os.listdir(dataset_folder) if f.endswith((".png", ".jpg"))]
    cleaned_ids = [f"clean-img-{i}" for i in range(len(cleaned_files))]
    cleaned_uris = [os.path.join(compare_folder, f) for f in cleaned_files]
    cleaned_embeddings = []

    for original_uri, cleaned_uri in zip(image_uris, cleaned_uris):
        try:
            clean_img = remove_text_from_image(original_uri)
            clean_np = cv2.cvtColor(np.array(clean_img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(cleaned_uri, clean_np)
            img = Image.open(cleaned_uri).convert('RGB').resize((224, 224))
            img_array = np.array(img)
            image_embedding = embedding_function([img_array])[0]
            cleaned_embeddings.append(image_embedding)
        except Exception as e:
            print(f"Error processing and embedding cleaned image from {original_uri}: {e}")
            cleaned_embeddings.append(None)

    valid_embeddings = [(uri, id, embed) for uri, id, embed in zip(cleaned_uris, cleaned_ids, cleaned_embeddings) if embed is not None]

    if valid_embeddings:
        uris_to_add, ids_to_add, embeddings_to_add = zip(*valid_embeddings)
        vectorstore_compare.add(
            uris=uris_to_add,
            ids=ids_to_add,
            embeddings=embeddings_to_add
        )
        print(f"Indexed {vectorstore_compare.count()} cleaned images in the compare database.")
    else:
        print("No valid cleaned images found to index.")

def get_coords(image, generated_text):
    h, w, _ = image.shape
    if "</point" in generated_text:
        matches = re.findall(
            r'<point\s+x="([\d.]+)"\s+y="([\d.]+)"(?:\s+alt="[^"]*")?>.*?</point>',
            generated_text
        )
        if len(matches) > 0:
            coordinates = [(int(float(x_val) / 100 * w), int(float(y_val) / 100 * h)) for x_val, y_val in matches]
        else:
            coordinates = []
    else:
        print("There are no points obtained from regex pattern")
        coordinates = []
    return coordinates

def custom_invoke_with_image(model, processor, image_path=None, closest_original_uri=None, max_length=50, temperature=0.7):
    try:
        prompt = (
            "Point to the object in the first image that corresponds to the most similar object in the second image. "
            "Respond with the coordinates using <point x=\"x\" y=\"y\"></point> format. Use percentage (0–100)."
        )

        query_img = Image.open(image_path).convert("RGB")
        retrieved_img = Image.open(closest_original_uri).convert("RGB")

        inputs = processor.process(text=prompt, images=[query_img, retrieved_img])
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        torch.cuda.empty_cache()

        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=max_length, temperature=temperature, stop_strings=["<|endoftext|>"]),
            tokenizer=processor.tokenizer
        )

        if isinstance(output, torch.Tensor) and output.ndim > 1 and output.size(1) > inputs['input_ids'].size(1):
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(response)
            return response
        else:
            return None

    except Exception as e:
        print(f"Error generating coordinates: {e}")
        return None

def search_cleaned_images(query_image_path, results=1):
    print(f"Searching for similar cleaned images to: {query_image_path}")
    try:
        clean_query_img = remove_text_from_image(query_image_path).convert('RGB').resize((224, 224))
        query_img_array = np.array(clean_query_img)
        query_embedding = embedding_function([query_img_array])
        if not query_embedding or len(query_embedding) == 0:
            raise ValueError("Embedding generation failed for query image.")
        results = vectorstore_compare.query(
            query_embeddings=query_embedding,
            n_results=results,
            include=["uris", "distances"]
        )
        return results
    except Exception as e:
        print(f"Error searching cleaned images: {e}")
        return None

def rag_image_query():
    image_files = [f for f in os.listdir(dataset_folder) if f.endswith((".png", ".jpg"))]
    image_ids = [f"img-{i}" for i in range(len(image_files))]
    image_uris = [os.path.join(dataset_folder, f) for f in image_files]

    index_original_images(image_files, image_ids, image_uris)
    index_cleaned_images(image_uris)

    while True:
        image_path = input("Enter the path to your query image (or 'exit' to quit): \n")
        if image_path.lower() == 'exit':
            break
        user_query = input("Enter your query related to the image: \n")

        cleaned_search_results = search_cleaned_images(image_path, results=1)

        if cleaned_search_results and "uris" in cleaned_search_results and cleaned_search_results["uris"]:
            closest_cleaned_uri = cleaned_search_results["uris"][0][0]
            closest_cleaned_distance = cleaned_search_results["distances"][0][0]
            print(f"Closest found cleaned image: {closest_cleaned_uri}")
            print(f"Distance (cleaned): {closest_cleaned_distance}")

            # Get the filename of the closest cleaned image
            closest_cleaned_filename = os.path.basename(closest_cleaned_uri)
            # Construct the path to the corresponding original image
            closest_original_uri = os.path.join(dataset_folder, closest_cleaned_filename)

            if os.path.exists(closest_original_uri):
                print(f"Corresponding original image: {closest_original_uri}")
                response = custom_invoke_with_image(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    closest_original_uri=closest_original_uri
                )

                if response:
                    img = cv2.imread(image_path)
                    coords = get_coords(img, response)
                    if coords:
                        draw_bounding_box_on_image(img, coords)
                        print(f"Bounding box coordinates: {coords}")
                    else:
                        print("Failed to extract coordinates from model output.")
                else:
                    print("No valid response from the model.")
            else:
                print("Could not find the corresponding original image for the closest cleaned image.")
        else:
            print("No similar cleaned images found.")

def main():
    rag_image_query()

if __name__ == "__main__":
    main()