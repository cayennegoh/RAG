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

warnings.filterwarnings("ignore")  # Suppress warnings globally

# Path to dataset
dataset_folder = "/pic/compare"

# Load the model
path = "/pic/"
model_id = "allenai/MolmoE-1B-0924"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=False
)

def load_model(model_id, quant_config: BitsAndBytesConfig = None, dtype="auto", device="cuda"):
    # Load the processor
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

# Setup ChromaDB with OpenCLIP
chroma_client = chromadb.PersistentClient(path="/pictures.db")
embedding_function = OpenCLIPEmbeddingFunction()

# Create collection in vectorstore
vectorstore = chroma_client.get_or_create_collection(
    "vectorstore",
    embedding_function=embedding_function,
    data_loader=ImageLoader()
)

# Add Image Embeddings with Captions
image_files = [f for f in os.listdir(dataset_folder) if f.endswith((".png", ".jpg"))]
image_ids = [f"img-{i}" for i in range(len(image_files))]
image_uris = [os.path.join(dataset_folder, f) for f in image_files]

image_metadatas = []
text_embeddings = []
image_embeddings = []

for uri in image_uris:
    try:
        img = Image.open(uri)
        extracted_caption = pytesseract.image_to_string(img).strip()
        print(extracted_caption)

        # Generate text embeddings
        text_embedding = embedding_function([extracted_caption])[0]
        text_embeddings.append(text_embedding)

        # Generate image embeddings (passing the image itself)
        img = img.convert('RGB')
        img = img.resize((224, 224))  # Resizing image to 224x224 for OpenCLIP model
        img_array = np.array(img)
        
        # Get image embedding
        image_embedding = embedding_function([img_array])[0]
        image_embeddings.append(image_embedding)

        image_metadatas.append({"caption": extracted_caption})
    except Exception as e:
        print(f"Error processing image {uri}: {e}")
        text_embeddings.append(None)
        image_embeddings.append(None)
        image_metadatas.append({"caption": "No caption extracted."})
print(image_metadatas)
print(image_uris)
# Store both text and image embeddings in the vectorstore
for i, image_id in enumerate(image_ids):
    existing_entry = vectorstore.get(ids=[image_id])
    if existing_entry and existing_entry['ids']:
        print()
    else:
        vectorstore.add(
            uris=image_uris,
            ids=image_ids,
            embeddings=text_embeddings,
            metadatas=image_metadatas
        )
        vectorstore.add(
            uris=image_uris,
            ids=image_ids,
            embeddings=image_embeddings,
            metadatas=image_metadatas
        )

print("Text and Image Embeddings added to the database.")
print(f"Total items in database: {vectorstore.count()}")

def query_db(query, results=10):
    """Query the database using embedded text."""
    print(f"Querying database with text: {query}")

    # Generate query embedding for the text
    query_embedding = embedding_function([query])[0]

    # Search in vectorstore
    results = vectorstore.query(
        query_embeddings=[query_embedding], 
        n_results=results, 
        include=["uris", "distances", "metadatas"]
    )
    return results
from torch.cuda.amp import autocast
def custom_invoke_with_image(model, processor, user_query=None, context=None,image_path=None, retrieved_path=None,max_length=150, temperature=0.7):#, image_embedding=None
    try:
        if any(word.lower() in context.lower() for word in user_query.split()):
            prompt = f"Identify the point which corresponds to the query'{user_query}'. If there is no relevant point, do not point" #f" Annotate the point in the query image '{image_path}' that corresponds most similar to the retrieved image '{retrieved_path}'."
        else:
            
            prompt = f'{user_query}'
        
        #     prompt = 
        #     # f" Analyze the query image and identify the point within it that visually corresponds most closely to the user query: '{user_query}'. Annotate this most relevant point"
            
        #     f"You are given reference information from a similar image: '{context}'. \n"
        #     f"Use this to enhance your understanding of the current query image.\n\n"
        #     f"Generate a **single, fluent** response that first provides a **short description** of the query image, followed by the reference image's context. "
        #     f"The description and context should be seamlessly combined into one sentence. "
        #     f"Do not mention 'context' or 'reference' — focus on describing the query image first."
        #     f"For example, if the reference caption says 'No phone usage,' and the query image shows a phone with a diagonal line, respond like:'The image shows a phone with a diagonal line across the screen, indicating no phone usage.'"
        #     f"Make the result feel like a human interpretation.\n"
        #     f"Keep it concise but clear. Combine the image context with the description and output only the query imagge\n "
        #     f"example: 'The image shows [vivid description], indicaating [meaning froom context]. DO not output the visual cueuse the word text. integrate it"
        # )
        # # input_text = f"Given the user query: '{user_query}', and a retrieved image with the following context: '{context}'. Analyze the query image and identify the point within it that visually corresponds most closely to the content described in the retrieved image's context. Annotate this most relevant point."
        # input_text = f"Given the following the user query: '{user_query}, If image context:'{context} does not contain user query, do not point find the point that matches most closely to the retrieved image annotate point that the user queries, taking into 'image context: '{context}'."#, and the prompt: '{prompt}', provide a description"#image '{image_embedding}'
        input_text = prompt#f"Given the user query: '{user_query}', and a retrieved image with the context: '{context}'. First, determine if the text of the context directly mentions any part of'{user_query}'. If it does, find the point in the query image that corresponds to '{user_query}' and annotate it. If the context does not directly mention any part of '{user_query}', then analyze the query image itself. If the query image visually contains '{user_query}', find the most prominent instance and annotate a representative point. If neither the context textually contains the query nor the query image visually contains it, do not annotate any point."
        image_path =  image_path # Replace with your image file
        img = Image.open(image_path)

        inputs = processor.process(text=input_text, images=[img])
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=max_length, temperature=temperature, stop_strings="<|endoftext|>" ),#[".", "!", "?"]
            tokenizer=processor.tokenizer
        )

        if isinstance(output, torch.Tensor) and output.ndim > 1 and output.size(1) > inputs['input_ids'].size(1):
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            response = "No valid response generated."

        return response.strip()
        print(response)

    except Exception as e:
        return f"Error generating response: {e}"

def image_to_image_search(image_path, results=1):
    print(f"Searching for similar images to: {image_path}")

    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img)

        embedding = embedding_function([img_array])

        if not embedding or len(embedding) == 0:
            raise ValueError("Embedding generation failed")

        results = vectorstore.query(
            query_embeddings=embedding,
            n_results=results,
            include=["uris", "distances", "metadatas"]
        )
        return results
    except Exception as e:
        print(f"Error processing the image: {e}")
        return None

def show_image_from_uri(uri):
    img = Image.open(uri)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
import cv2
def read_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found or cannot be read from path: {path}")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image
# def rag_image_query():
#     """Retrieves relevant image data and generates a response using LLM."""

#     while True:
#         image_path = input("Enter the path to your query image (or 'exit' to quit): \n")
        
#         if image_path.lower() == 'exit':
#             break
#         user_query = input("Enter your query related to the image: \n")

#         image_search_results = image_to_image_search(image_path, results=1)

#         if image_search_results and "metadatas" in image_search_results and image_search_results["metadatas"][0]:
#             top_result_metadata = image_search_results["metadatas"][0][0] if image_search_results["metadatas"][0] else None

#             if "uris" in image_search_results and image_search_results["uris"]:
#                 closest_image_uri = image_search_results["uris"][0][0]
#                 closest_image_distance = image_search_results["distances"][0][0]
#                 print(f"Closest found image: {closest_image_uri}")
#                 print(f"Distance: {closest_image_distance}")

#                 if closest_image_distance > 16:
#                     print("No relevant image context found.")
                    

#             if top_result_metadata and "caption" in top_result_metadata:
#                 context = top_result_metadata["caption"]

#                 if not context.strip():
#                     print("No relevant caption found for the closest image.")
#                     return

#                 try:
#                     response = custom_invoke_with_image(
#                         model=model, 
#                         processor=processor, 
#                         user_query=user_query,
#                         context=context,
#                         image_path=image_path
#                     )
#                     print(f"LLM Response: {response}")
#                     image=read_image(image_path)
#                     get_coords(image,response)
                    
                    
#                 except Exception as e:
#                     print(f"Error generating response: {e}")
#             else:
#                 print("No caption available for the closest image.")
#         else:
#             print("No relevant image context found.")
def rag_image_query():
    """Retrieves multiple relevant images and generates response points on the query image using the LLM."""
    while True:
        image_path = input("Enter the path to your query image (or 'exit' to quit): \n")
        
        if image_path.lower() == 'exit':
            break
        user_query = input("Enter your query related to the image: \n")

        TOP_K = 1  # Number of similar images to fetch
        image_search_results = image_to_image_search(image_path, results=TOP_K)

        if image_search_results and "metadatas" in image_search_results and image_search_results["metadatas"]:
            all_coords = []
            print(f"Found {len(image_search_results['metadatas'][0])} similar images")

            image = read_image(image_path)  # Read only once

            for i in range(len(image_search_results['metadatas'][0])):
                metadata = image_search_results["metadatas"][0][i]
                distance=image_search_results["distances"][0][i]
                if distance > 16:
                    print(f"\n#{i+1} Similar Image is too far (Distance: {distance:.2f}). Skipping annotation.")
                    continue
                if not metadata or "caption" not in metadata or not metadata["caption"].strip():
                    continue

                context = metadata["caption"]
                uri = image_search_results["uris"][0][i]
                distance = image_search_results["distances"][0][i]
                print(f"\n#{i+1} Similar Image: {uri}\nDistance: {distance:.2f}\nCaption: {context}\n")

                try:
                    response = custom_invoke_with_image(
                        model=model, 
                        processor=processor, 
                        user_query=user_query,
                        context=context,
                        image_path=image_path,
                        retrieved_path=uri
                    )
                    print(f"LLM Response: {response}")

                    coords = get_coords(image.copy(), response)  # Copy to avoid drawing multiple times on the same image
                    all_coords.extend(coords)
                except Exception as e:
                    print(f"Error generating response for image #{i+1}: {e}")

            if all_coords:
                overlay_points_on_image(image, all_coords)
                print(f"Overlay complete. Total points: {len(all_coords)}")
            else:
                print("No coordinates found for any matched images.")
        else:
            print("No relevant image context found.")

import re
def overlay_points_on_image(image, points, radius=5, color=(255, 0, 0)):
 
    # Define the BGR color equivalent of the hex color #f3599e
    pink_color = (158, 89, 243)  # Color for the points (BGR format)
 
    for (x, y) in points:
        # Draw an outline for the point
        outline = cv2.circle(
            image, 
            (int(x), int(y)), 
            radius=radius + 1, 
            color=(255, 255, 255), 
            thickness=2, 
            lineType=cv2.LINE_AA
        )
        # Draw the point itself
        image_pt = cv2.circle(
            outline, 
            (int(x), int(y)), 
            radius=radius, 
            color=color, 
            thickness=-1, 
            lineType=cv2.LINE_AA
        )
     
    # Save and convert the image
    sav_image = image_pt.copy()
    image = cv2.cvtColor(sav_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("/pic/test/output_pt.jpg", sav_image)
def get_coords(image, generated_text):
    h, w, _ = image.shape
 
    if "</point" in generated_text:
        matches = re.findall(
            r'(?:x(?:\d*)="([\d.]+)"\s*y(?:\d*)="([\d.]+)")', generated_text
        )
        if len(matches) > 1:
            coordinates = [
                (int(float(x_val) / 100 * w), int(float(y_val) / 100 * h))
                for x_val, y_val in matches
            ]
        else:
            coordinates = [
                (int(float(x_val) / 100 * w), int(float(y_val) / 100 * h))
                for x_val, y_val in matches
            ]
 
    else:
        print("There are no points obtained from regex pattern")
 
    overlay_points_on_image(image,coordinates)
    return coordinates

def main():
    rag_image_query()

if __name__ == "__main__":
    main()
