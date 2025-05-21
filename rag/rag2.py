path = "/pic/"

#                       )
import os
import numpy as np
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
from langchain_experimental.open_clip import OpenCLIPEmbeddings
import numpy as np
from PIL import Image as _PILImage
import uuid
import faiss
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

embedding_function = OpenCLIPEmbeddings()
image_uris = sorted(
    [
        os.path.join(path, image_name)
        for image_name in os.listdir(path)
        if image_name.endswith("food.jpg")
    ]
)

# Create embeddings for the images
image_embeddings = []
for image_uri in image_uris:
    # Embed the image
    embedding = embedding_function.embed_image([image_uri])
    print(image_uri)
    # Flatten the embedding if it's nested more than necessary
    if isinstance(embedding[0], list):  # If it's a 2D list, flatten it to 1D
        embedding = [item for sublist in embedding for item in sublist]  # Flatten
    
    # Ensure the embedding is 1D (list of floats)
    image_embeddings.append(embedding)
print(image_embeddings[:1])
# Add images along with embeddings
image_documents = [Document(page_content=uri, metadata={"type": "image"}) for uri in image_uris]


# Combine text and image documents
all_documents = image_documents 
all_embeddings = image_embeddings
image_embeddings=np.array(image_embeddings, dtype=np.float32)
dimension=1024
index = faiss.IndexFlatL2(dimension)
index.add(image_embeddings)
# Create FAISS index
vectorstore = FAISS.from_documents(all_documents, embedding_function)
# Initialize OpenCLIP embeddings
retriever = vectorstore.as_retriever()
# Function to get embeddings for images
import base64
import io
from io import BytesIO

import numpy as np
from PIL import Image


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string.

    Args:
    base64_string (str): Base64 string of the original image.
    size (tuple): Desired size of the image as (width, height).

    Returns:
    str: Base64 string of the resized image.
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_base64(s):
    """Check if a string is Base64 encoded"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False


def split_image_text_types(docs):
    """Split numpy array images and texts"""
    images = []
    text = []
    for doc in docs:
        doc = doc.page_content  # Extract Document contents
        if is_base64(doc):
            # Resize image to avoid OAI server error
            images.append(
                resize_base64_image(doc, size=(250, 250))
            )  # base64 encoded str
        else:
            text.append(doc)
    return {"images": images, "texts": text}

# query = "Sample query to test retriever"
# retrieved_docs = retriever.retrieve(query)
# for doc in retrieved_docs:
#     print(doc)

from operator import itemgetter

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI


openai_api_key = os.environ["OPENAI_API_KEY"]
def prompt_func(data_dict):
    # Joining the context texts into a single string
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
            },
        }
        messages.append(image_message)

    # Adding the text message for analysis
    text_message = {
        "type": "text",


        "text": (
            "As an expert art critic and historian, your task is to analyze and interpret images, "
            "considering their historical and cultural significance. Alongside the images, you will be "
            "provided with related text to offer context. Both will be retrieved from a vectorstore based "
            "on user-input keywords. Please use your extensive knowledge and analytical skills to provide a "
            "comprehensive summary that includes:\n"
            "- A detailed description of the visual elements in the image.\n"
            "- The historical and cultural context of the image.\n"
            "- An interpretation of the image's symbolism and meaning.\n"
            "- Connections between the image and the related text.\n\n"
            f"User-provided keywords: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)

    return [HumanMessage(content=messages)]


import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
   


def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device_map()

print("I am running on: " + device)
# processor = AutoProcessor.from_pretrained(
#     'allenai/Molmo-7B-O-0924',
#     trust_remote_code=True,
#     use_fast=False,
#     torch_dtype='auto',
#     #device_map = device
#     device_map = "cpu"
#     #device_map = "cuda" if torch.cuda.is_available() else device_map == "cpu"
# )

# model = AutoModelForCausalLM.from_pretrained(
#     'allenai/Molmo-7B-O-0924',
#     trust_remote_code=True,
#     torch_dtype='auto',
#     low_cpu_mem_usage = True,
#     #device_map = device
#     device_map = "cpu"
#     #device_map = "cuda" if torch.cuda.is_available() else device_map == "cpu"
# )


# def generate_with_molmo(data):
#     inputs = processor.process(
#         images=[Image.open(io.BytesIO(base64.b64decode(data["image"])))] if data["image"] else None,
#         text=data["text"]
#     )


# query = "Sample query to test retriever"
# retrieved_docs = retriever.retrieve(query)
# for doc in retrieved_docs:
#     print(doc)
#     output = model.generate_from_batch(
#         inputs,
#         GenerationConfig(max_new_tokens=500, stop_strings=["<|endoftext|>"]),
#         tokenizer=processor.tokenizer
#     )

#     generated_tokens = output[0, inputs['input_ids'].size(1):]
#     generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
#     return generated_text# RAG pipeline


model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)
chain = (
    {
        "context": retriever | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(prompt_func)
    | model #RunnableLambda(generate_with_molmo)
    | StrOutputParser()
)
# from langchain_core.prompts import ChatPromptTemplate
# prompt = ChatPromptTemplate.from_messages(
#     [("user", "Tell me a joke")],
# )
# chain = prompt | model | StrOutputParser()
# chain.invoke({"adjective": "funny"})
from IPython.display import HTML, display


def plt_img_base64(img_base64):
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'

    # Display the image by rendering the HTML
    display(HTML(image_html))

docs=retriever.invoke("Woman with children", k=10)
print("Retrieved docs: ", docs)
for doc in docs:
    if is_base64(doc.page_content):
        plt_img_base64(doc.page_content)
    else:
        print(doc.page_content)
# user_question = "Woman with children"
# question_embedding = embedding_function.encode(user_question)  # Assuming the embedding model is defined
# question_embedding = np.expand_dims(question_embedding, axis=0)  # Shape should be (1, d)

# input_data = {
#     "context": retriever,
#     "question" : question_embedding
# }

# response = chain.invoke(input_data)
image_embedding=np.expand_dims(image_embeddings, axis=0)
query = "What objects are present?"
image_embedding = embedding_function.embed_image(["/pic/figure-32-3.jpg"])  # Define this function
import numpy as np
# Assuming embedding_function is already initialized
image_embedding = np.array(image_embedding).squeeze()
print(image_embedding.shape)
question_embedding = embedding_function.embed_query("What objects are present?")  # Embed the question
question_embedding = np.array(question_embedding, dtype=np.float32).reshape(1, -1)  # Ensure it's a 2D numpy array

# Now, you can pass the question_embedding into the similarity search
retriever = vectorstore.as_retriever()
# retrieved_docs=vectorstore.similarity_search(query=question_embedding, k=10)
# for doc in retrieved_docs:
#     print(doc.page_content)
# k=5
# scores, indices = index.search(query, k)
# print("scores", scores)
# print("indices", indices)
# print("Image", image_embedding.shape)
docs=retriever.invoke("Woman with children", k=10)
print("retrieved docs: ", docs)

#reponse = chain.apply({"question": query, "context": contexts})
response = chain.invoke({"question": query, "context": image_embedding})

print(response)

# print(response)
#chain.invoke({"question": "Woman with children"})



