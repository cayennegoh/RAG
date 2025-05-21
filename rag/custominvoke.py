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
from transformers import AutoModelForCausalLM, AutoProcessor
warnings.filterwarnings("ignore")
image_loader = ImageLoader()

path = "/pic/"
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    use_fast=False,
    torch_dtype='auto',
    #device_map = device
    device_map = "cpu"
    #device_map = "cuda" if torch.cuda.is_available() else device_map == "cpu"
)

model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    low_cpu_mem_usage = True,
    #device_map = device
    device_map = "cpu"
    #device_map = "cuda" if torch.cuda.is_available() else device_map == "cpu"
)
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    use_fast=False,
    torch_dtype='auto',
    #device_map = device
    device_map = "cpu"
    #device_map = "cuda" if torch.cuda.is_available() else device_map == "cpu"
)

model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    low_cpu_mem_usage = True,
    #device_map = device
    device_map = "cpu"
    #device_map = "cuda" if torch.cuda.is_available() else device_map == "cpu"
)

# 2. Setup ChromaDB with OpenCLIP
chroma_client = chromadb.PersistentClient(path="/pict.db")
embedding_function = OpenCLIPEmbeddingFunction()

vectorstore = chroma_client.get_or_create_collection(
    "vectorstore",
    embedding_function=embedding_function,
    data_loader=image_loader
)

# Clear the database if it exists, to avoid duplicate ent


# Add Image Embeddings with Captions
dataset_folder = "/pic/dataset"

image_files = [f for f in os.listdir(dataset_folder) if f.endswith((".png", ".jpg"))]

image_ids = [f"img-{i}" for i in range(len(image_files))]
image_uris = [os.path.join(dataset_folder, f) for f in image_files]

# Extract captions and generate text embeddings
# Add Image Embeddings with Captions and Image Embeddings
image_metadatas = []
text_embeddings = []
image_embeddings = []  # Add a list to store image embeddings

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
    embeddings=text_embeddings,  # You can also store image embeddings here if you want
    metadatas=image_metadatas
)
vectorstore.add(
    uris=image_uris,
    ids=image_ids,
    embeddings=image_embeddings,  # You can also store image embeddings here if you want
    metadatas=image_metadatas
)

print("Text and Image Embeddings added to the database.")
print(f"Total items in database: {vectorstore.count()}")

# === Functions for Querying the VectorDB ===
def query_db(query, results=1):
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

def get_llm_response(caption):
    """Passes the caption to the LLM and returns the response."""
    vision_model = model
    prompt = f"Based on the following caption: '{caption}', provide a yes or no answer."
    try:
        response = custom_invoke_with_image(
            model=model, 
            processor=processor, 
             # Or use image_embedding if available
            user_query=user_query,
            context=context,
            prompt=prompt
            
        )
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
import torch
import numpy as np
from PIL import Image
import torch
import numpy as np
from PIL import Image
from transformers import GenerationConfig
import torch
from transformers import GenerationConfig

def custom_invoke_with_image(model, processor, user_query=None, context=None, prompt=None, max_length=512, temperature=0.7):
    try:
        # if prompt:
        #     inputs = prompt.format(context=context, user_query=user_query)
        if context:
            # inputs = processor.process(text=f"{context} {user_query}")
            input_text = f"Given the following image context: '{context}' and the user query: '{user_query}', provide a yes or no answer in one word."
        else:
            inputs = processor.process(text=user_query)
        inputs=processor.process(text=input_text)
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
    vision_model = model(model="gpt-4o", temperature=0.0)
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

        response = custom_invoke_with_image(
            model=model, 
            processor=processor, 
              # Or use image_embedding if available
            user_query=user_query
        )       
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

#         llm = model(model="gpt-4o", temperature=0.0)
#         prompt = f"Given the following image context: '{context}' and the user query: '{user_query}', provide a detailed response."
        
#         try:
#             response = custom_invoke_with_image(
#     model=model, 
#     processor=processor, 
#     image_path=image_path,  # Or use image_embedding if available
#     user_query=user_query
# )
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

            llm = model
            prompt = f"Given the following image context: '{context}' and the user query: '{user_query}', provide a yes or no answer. In one word."

            try:
                response = custom_invoke_with_image(
                    model=model, 
                    processor=processor, 
                  # Or use image_embedding if available
                    user_query=user_query,
                    context=context,
                    prompt=prompt
                )
                return response
            except Exception as e:
                return f"Error generating response: {e}"
        else:
            print("No caption available for the closest image.")
            return "No relevant image context found."
    else:
        return "No relevant image context found."
def main():
    response = rag_image_query()
    print("\nRAG Response:\n", response)

if __name__ == "__main__":
    main()

