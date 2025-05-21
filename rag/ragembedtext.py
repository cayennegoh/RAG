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
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig
import torch

warnings.filterwarnings("ignore")
image_loader = ImageLoader()

# Paths and model configuration
path = "/pic/dataset"
model_id = "allenai/MolmoE-1B-0924"
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=False)

# Load model and processor
def load_model(model_id, quant_config: BitsAndBytesConfig = None, dtype="auto", device='cpu'):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto", device_map='cpu')
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype, device_map='cpu', quantization_config=quant_config)
    return model, processor

model, processor = load_model(model_id, quant_config=quant_config, dtype="auto", device='cpu')

# Setup ChromaDB with OpenCLIP
chroma_client = chromadb.PersistentClient(path="/picts.db")
embedding_function = OpenCLIPEmbeddingFunction()
vectorstore = chroma_client.get_or_create_collection("vectorstore", embedding_function=embedding_function, data_loader=image_loader)

# Add Image Embeddings with Captions and Image Embeddings
dataset_folder = "/pic/dataset"
image_files = [f for f in os.listdir(dataset_folder) if f.endswith((".png", ".jpg"))]
image_ids = [f"img-{i}" for i in range(len(image_files))]
image_uris = [os.path.join(dataset_folder, f) for f in image_files]

# Extract captions and generate text & image embeddings
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
        img = img.resize((224, 224))  # Resizing image for OpenCLIP model
        img_array = np.array(img)
        
        # Get image embedding
        image_embedding = embedding_function([img_array])[0]  # Assuming OpenCLIP returns a list
        image_embeddings.append(image_embedding)

        image_metadatas.append({"caption": extracted_caption})
    except Exception as e:
        print(f"Error processing image {uri}: {e}")
        text_embeddings.append(None)
        image_embeddings.append(None)
        image_metadatas.append({"caption": "No caption extracted."})

# Store both text and image embeddings in the vectorstore
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

# === Functions for Querying the VectorDB ===
def query_db(query, results=1):
    """Query the database using embedded text."""
    print(f"Querying database with text: {query}")
    query_embedding = embedding_function([query])[0]

    results = vectorstore.query(
        query_embeddings=[query_embedding], 
        n_results=results, 
        include=["uris", "distances", "metadatas"]
    )
    return results

def get_llm_response(caption, user_query, image_embedding):
    """Passes the caption, user query, and image to the LLM and returns the response."""
    try:
        prompt = f"Given the image context: '{caption}', the user query: '{user_query}', provide a detailed response."
        response = custom_invoke_with_image(
            model=model, 
            processor=processor, 
            user_query=user_query,
            context=caption,
            image_embedding=image_embedding,
            
        )
        return response
    except Exception as e:
        return f"Error getting LLM response: {e}"

import requests
# Function for querying the image and generating response
def custom_invoke_with_image(model, processor, user_query, context, image,image_embedding,  max_length=512, temperature=0.7):
    try:
        input_text = f"Given the following image context: '{context}' and the user query: '{user_query}',generate a response ."
        img = Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)

        inputs = processor.process(text=input_text, images=[img])
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=max_length, temperature=temperature, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        if isinstance(output, torch.Tensor) and output.ndim > 1 and output.size(1) > inputs['input_ids'].size(1):
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            response = "No valid response generated."

        return response.strip()

    except Exception as e:
        return f"Error generating response: {e}"

import numpy as np
from PIL import Image
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

def generate_image_embedding(image_path):
    """Generates the embedding of the query image."""
    try:
        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((224, 224))  # Resize for consistent input size
        img_array = np.array(img)

        # Use the OpenCLIPEmbeddingFunction to generate the image embedding
        embedding_function = OpenCLIPEmbeddingFunction()  # Assuming this is your image embedding function
        image_embedding = embedding_function([img_array])[0]  # Get the embedding for the image

        return image_embedding
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def image_to_image_search(query_image_embedding, results=1):
    """Search for similar images using the query image embedding."""
    try:
        # Query the vector store to find the closest images using the image embedding
        search_results = vectorstore.query(
            query_embeddings=[query_image_embedding],  # Pass in the query image embedding
            n_results=results, 
            include=["uris", "distances", "metadatas"]
        )
        return search_results
    except Exception as e:
        print(f"Error in image search: {e}")
        return None

def rag_image_query():
    """Retrieves relevant image data and generates a response using LLM."""
    # Take user input for image path and query
    image_path = input("Enter the path to your query image: \n")
    user_query = input("Enter your query related to the image: \n")

    # Step 1: Get the embedding for the query image
    query_image_embedding = generate_image_embedding(image_path)
    # if not query_image_embedding.any():
    #     return "Error generating image embedding."

    # Step 2: Perform the image search with the query image embedding
    image_search_results = image_to_image_search(query_image_embedding, results=1)

    if image_search_results and "metadatas" in image_search_results and image_search_results["metadatas"][0]:
        # Extract the metadata for the top result
        top_result_metadata = image_search_results["metadatas"][0][0]
        
        if top_result_metadata and "caption" in top_result_metadata:
            context = top_result_metadata["caption"]

            if not context.strip():
                print("No relevant caption found for the closest image.")
                return "No relevant image context found."

            # Step 3: Now you can proceed with RAG and get the LLM response
            llm_response = custom_invoke_with_image(
                model=model, 
                processor=processor, 
                user_query=user_query,
                context=context,
                image=image_path,
                
                image_embedding=query_image_embedding  # Pass the query image embedding here
            )
            return llm_response

    else:
        return "No relevant image context found."
def main():
    response = rag_image_query()
    print("\nRAG Response:\n", response)

if __name__ == "__main__":
    main()
