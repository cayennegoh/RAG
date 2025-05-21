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
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings("ignore")
image_loader = ImageLoader()

path = "/pic/"

# 2. Setup ChromaDB with OpenCLIP
chroma_client = chromadb.PersistentClient(path="/pict.db")
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
dataset_folder = "/pic/dataset"

image_files = [f for f in os.listdir(dataset_folder) if f.endswith((".png", ".jpg"))]

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

vectorstore.add(
    uris=image_uris,
    ids=image_ids,
    metadatas=image_metadatas
)

print("Text and Images added to the database.")
print(f"Total items in database: {vectorstore.count()}")

# === Functions for Querying the VectorDB ===
def query_db(query, results=1):
    print(f"Querying the database for: {query}")
    results = vectorstore.query(
        query_texts=[query], n_results=results, include=["uris", "distances", "metadatas"]
    )
    return results

def get_llm_response(caption):
    """Passes the caption to the LLM and returns the response."""
    vision_model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    low_cpu_mem_usage = True,
    #device_map = device
    device_map = "cpu"
    
    )
    prompt = f"Based on the following caption: '{caption}', provide a detailed response."
    try:
        response = vision_model.__call__(prompt).content
        return response
    except Exception as e:
        return f"Error getting LLM response: {e}"

def print_results(results):
    if results and "uris" in results and results["uris"] and "ids" in results and results["ids"]:
        for idx, uri in enumerate(results["uris"][0]):
            print(f"ID: {results['ids'][0][idx]}")
            print(f"Distance: {results['distances'][0][idx]}")
            print(f"Path: {uri}")
            
            # Check if metadatas exists and has values
            if results.get("metadatas") and results["metadatas"][0] and len(results["metadatas"][0]) > idx:
                metadata = results["metadatas"][0][idx]
                # Now check if caption exists in metadata
                if "caption" in metadata:
                    caption = metadata["caption"]
                    print(f"Caption: {caption}")
                    llm_response = get_llm_response(caption)
                    print(f"LLM Response: {llm_response}")
                else:
                    print("No caption available.")
            
            show_image_from_uri(uri)
            print("\n")
    else:
        print("No results found.")

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

def describe_image(image_path):
    vision_model = ChatOpenAI(model="gpt-4o", temperature=0.0)
    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the contents of this image in detail.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":f"data:image/jpeg;base64,{image_data}",
                        },
                    },
                ],
            },
        ]

        response = vision_model.__call__(prompt).content
        return response
    except Exception as e:
        return f"Error describing image: {e}"

# def rag_image_query():
#     """Retrieves relevant image data and generates a response using LLM."""
    
#     image_path = input("Enter the path to your query image: \n")
#     user_query = input("Enter your query related to the image: \n")

#     image_search_results = image_to_image_search(image_path, results=1)

#     if image_search_results and "metadatas" in image_search_results and image_search_results["metadatas"][0]:
#         context = " ".join(
#             metadata["caption"] for metadata in image_search_results["metadatas"][0] if "caption" in metadata
#         )
#         if "uris" in image_search_results and image_search_results["uris"]:
#             closest_image_uri = image_search_results["uris"][0][0]  # Closest image URI
#             closest_image_distance = image_search_results["distances"][0][0]  # Distance for the closest image
#             print(f"Closest found image: {closest_image_uri}")
#             print(f"Distance: {closest_image_distance}")

#         if not context.strip():
#             print("No relevant captions found in database.")
#             return "No relevant image context found."

#         llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
#         prompt = f"Given the following image context: '{context}' and the user query: '{user_query}', provide a detailed response."
        
#         try:
#             response = llm.__call__(prompt).content
#             return response
#         except Exception as e:
#             return f"Error generating response: {e}"
#     else:
#         return "No relevant image context found."
def rag_image_query():
    """Retrieves relevant image data and generates a response using LLM."""

    image_path = input("Enter the path to your query image: \n")
    user_query = input("Enter your query related to the image: \n")

    image_search_results = image_to_image_search(image_path, results=1)

    if image_search_results and "metadatas" in image_search_results and image_search_results["metadatas"][0]:
        # Access the metadata of the top result directly
        top_result_metadata = image_search_results["metadatas"][0][0] if image_search_results["metadatas"][0] else None

        if top_result_metadata and "caption" in top_result_metadata:
            context = top_result_metadata["caption"]
            if "uris" in image_search_results and image_search_results["uris"]:
                closest_image_uri = image_search_results["uris"][0][0]  # Closest image URI
                closest_image_distance = image_search_results["distances"][0][0]  # Distance for the closest image
                print(f"Closest found image: {closest_image_uri}")
                print(f"Distance: {closest_image_distance}")

            if not context.strip():
                print("No relevant caption found for the closest image.")
                return "No relevant image context found."

            from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = 'allenai/Molmo-7B-O-0924'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype='auto',
    low_cpu_mem_usage=True,
    device_map="cpu"  # Or 'auto' if you want to use GPU
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
import torch
def generate_response(context, user_query):
    # Combine the context and query into the prompt
    prompt = f"Given the following image context: '{context}' and the user query: '{user_query}', provide a detailed response."

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    input_ids=input_ids[:, 35]
    attention_mask = inputs["attention_mask"]
    attention_mask=attention_mask[:, 35]
    position_ids = torch.arange(0, inputs['input_ids'].shape[-1], device=model.device).unsqueeze(0)
    if input_ids.dim() == 1:  
        input_ids = input_ids.unsqueeze(0)  # Expand to (1, seq_length)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)  # Expand to (1, seq_length)

    print("input", input_ids.shape)
    print(attention_mask.shape)
    # inputs['position_ids'] = position_ids
    # print(inputs.keys())  # Check what keys are available
    print(position_ids.shape)  # Ensure shapes match expectations
    # print(inputs.get('position_ids', 'Not available'))  # Check for position_ids existence
    # inputs_tensor = None
    # if inputs_tensor.ndim == 1:
    #     inputs_tensor = inputs_tensor.unsqueeze(0)  # Convert [seq_length] -> [1, seq_length]
    # if inputs_tensor is None or inputs_tensor.numel() == 0:
    #     raise ValueError("Error: `inputs_tensor` is empty or uninitialized!")
   
    # print(type(inputs))
    # print(inputs.keys() if isinstance(inputs, dict) else inputs.shape)
    # inputs.pop('position_ids', None)  # Remove position_ids from the input
    # model_output = model(**inputs)  # Pass inputs without position_ids
    # outputs=model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=200, num_return_sequences=1)
    # print(outputs)
    try:
        # Generate a response
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,position_ids=position_ids,max_length=200, num_return_sequences=1)
        # outputs = model (**iniputs)
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("input_ids shape:", input_ids.shape)
        print("attention_mask shape:", attention_mask.shape)
        print("position_ids shape:", position_ids.shape if 'position_ids' in inputs else "Not used")
        if 'position_ids' in locals():
            print("position_ids shape: ", position_ids.shape)
        
        return response
    except Exception as e:
        return f"Error generating response: {e}"

# Example usage


def main():
    context = "An image showing a sunset over the ocean."
    user_query = "What is the atmosphere like in the image?"
    response = generate_response(context, user_query)
    print(response)

if __name__ == "__main__":
    main()
