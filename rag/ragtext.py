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
dataset_folder = "/pic/dataset"

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
chroma_client = chromadb.PersistentClient(path="/pct.db")
embedding_function = OpenCLIPEmbeddingFunction()
image_collection = chroma_client.get_or_create_collection(
    "image_embeddings",
    embedding_function=embedding_function,
    data_loader=ImageLoader()
)

text_collection = chroma_client.get_or_create_collection(
    "text_embeddings",
    embedding_function=embedding_function
)
captions = ["Road sign of a tree", "Cannot use phones", "Turtle crossing"]
# Add Image Embeddings with Captions
image_files = sorted([f for f in os.listdir(dataset_folder) if f.endswith((".png", ".jpg"))])
image_ids = [f"img-{i}" for i in range(len(image_files))]
image_uris = [os.path.join(dataset_folder, f) for f in image_files]

# Instead of saving raw captions, save their embeddings
image_metadatas = []
text_embeddings = []
image_embeddings = []
for img_id, uri, caption in zip(image_ids, image_uris, captions):
    try:
        img = Image.open(uri).convert('RGB').resize((224, 224))
        img_array = np.array(img)

        # Generate embedding for the caption
        text_embedding = embedding_function([caption])[0]
        
        # Save the caption embedding instead of the raw text
        text_collection.add(
            ids=[f"{img_id}-text"],
             # The original caption text is still stored here, if you want
            embeddings=[text_embedding],  # Store the embedding instead of raw text
            metadatas=[{"type": "text", "img_id": img_id, "caption" :caption}]
        )

        # Add image embedding to the image collection
        image_embedding = embedding_function([img_array])[0]
        image_collection.add(
            ids=[f"{img_id}-image"],
            uris=[uri],
            embeddings=[image_embedding],
            metadatas=[{"type": "image", "img_id": img_id}]  # Also store caption embedding here if needed
        )
        result = text_collection.get(ids=[f"{img_id}-text"], include=["embeddings", "metadatas"])
        print(f"Stored text embedding for {img_id}: {result}")
        print(f"Added {img_id} to DB with caption embedding.")
    except Exception as e:
        print(f"Error processing {uri}: {e}")

# At this point, your text embeddings are saved in ChromaDB instead of raw text.

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
from torch.cuda.amp import autocast
def custom_invoke_with_image(model, processor, user_query=None, context=None,image_path=None, max_length=150, temperature=0.7):#, image_embedding=None
    try:
        prompt = input_text = (
            
            f"You are given reference information from a similar image: '{context}'. \n"
            # f"Use this to enhance your understanding of the current query image.\n\n"
            # f"Generate a **single, fluent** response that first provides a **short description** of the query image, followed by the reference image's context. "
            # f"The description and context should be seamlessly combined into one sentence. "
            # f"Do not mention 'context' or 'reference' — focus on describing the query image first."
            # f"For example, if the reference caption says 'No phone usage,' and the query image shows a phone with a diagonal line, respond like:'The image shows a phone with a diagonal line across the screen, indicating no phone usage.'"
            # f"Make the result feel like a human interpretation.\n"
            # f"Keep it concise but clear. Combine the image context with the description and output only the query imagge\n "
            f"example: 'The image shows [vivid description], indicaating '{context}'. DO not output the visual cueuse the word text. integrate it"
        )

        input_text = f"Given the following image context: '{context}' and the user query: '{user_query}', and the prompt: '{prompt}', provide a description"#image '{image_embedding}'
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

    except Exception as e:
        return f"Error generating response: {e}"

def image_to_image_search(image_path, results=1):
    print(f"Searching for similar images to: {image_path}")

    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB').resize((224, 224))
            img_array = np.array(img)

        embedding = embedding_function([img_array])
        results = image_collection.query(
            query_embeddings=embedding,
            n_results=results,
            include=["uris", "distances"]
        )

        if results and "ids" in results:
            top_img_id = results["ids"][0][0]  # e.g., img-0-image
            img_id_base = top_img_id.split("-image")[0]
            print(f"Top match image ID: {img_id_base}")
            return img_id_base
        return None
    except Exception as e:
        print(f"Error processing the image: {e}")
        return None
def get_text_embedding_from_image_id(img_id):
    try:
        result = text_collection.get(
            ids=[f"{img_id}-text"],
            include=["embeddings", "metadatas"]
            
        )
        print(f"Retrieved result for {img_id}: {result}")
        if (
            result is not None
            and "embeddings" in result
            
            and len(result["embeddings"]) > 0
        ):
            caption = result["metadatas"][0].get("caption", "No caption found")
            return result["embeddings"][0], caption
#         else: result["documents"][0]  
            print(f"No text found for {img_id}")
            return None
    except Exception as e:
        print(f"Error retrieving text embedding: {e}")
        return None

def rag_image_query():
    

    """Retrieves relevant image data and generates a response using LLM."""

    # Loop for multiple queries
    while True:
        image_path = input("Enter the path to your query image (or 'exit' to quit): \n")
        
        if image_path.lower() == 'exit':
            break
        user_query = input("Enter your query related to the image: \n")

        matched_img_id = image_to_image_search(image_path, results=1)
        if not matched_img_id:
            print("No similar image found.")
            continue

        context_embedding, caption = get_text_embedding_from_image_id(matched_img_id)
        print(caption)
        if context_embedding is None:
            print("No associated caption embedding found.")
            continue

        context = "Retrieved context from similar image."  # You can use the caption or description here too

        try:
            response = custom_invoke_with_image(
                model=model,
                processor=processor,
                user_query=user_query,
                context=caption,
                image_path=image_path
            )
            print(f"LLM Response: {response}")
        except Exception as e:
            print(f"Error generating response: {e}")

def main():
    rag_image_query()

if __name__ == "__main__":
    main()
