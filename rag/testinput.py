import os
import openai
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from blip_model import extract_image_details
from gpt_model import generate_response
from unstructured.partition.pdf import partition_pdf
import chromadb
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
import base64
import io
from io import BytesIO
import numpy as np
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize session state for image details and messages
if "image_details" not in st.session_state:
    st.session_state.image_details = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Image and PDF RAG with Chat")

# File uploader for image upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

st.session_state.uploaded_file = uploaded_file

st.session_state.messages = []
        # Show loading status while processing the image
with st.status("Processing image...", state="running"):
    image = Image.open(uploaded_file)
    st.session_state.image_details = extract_image_details(image)

st.success("Image processed successfully.")


path = "/pic/"  # Adjust path as needed

# Extract images, tables, and chunk text from PDF
raw_pdf_elements = partition_pdf(
    filename=path + "photo.pdf",  # Adjust filename as needed
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path,
)
tables = []
texts = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        tables.append(str(element))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        texts.append(str(element))

# Create chroma vectorstore
vectorstore = Chroma(
    collection_name="mm_rag_clip_photos", embedding_function=OpenCLIPEmbeddings()
)

# Get image URIs with .jpg extension only
image_uris = sorted(
    [
        os.path.join(path, image_name)
        for image_name in os.listdir(path)
        if image_name.endswith(".jpg")
    ]
)

# Add images to vectorstore
vectorstore.add_images(uris=image_uris)

# Make retriever
retriever = vectorstore.as_retriever()

# Helper functions
def resize_base64_image(base64_string, size=(128, 128)):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def is_base64(s):
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False

def split_image_text_types(docs):
    images = []
    text = []
    for doc in docs:
        doc = doc.page_content
        if is_base64(doc):
            images.append(resize_base64_image(doc, size=(250, 250)))
        else:
            text.append(doc)
    return {"images": images, "texts": text}

def prompt_func(data_dict):
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []
    if data_dict["context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
            },
        }
        messages.append(image_message)
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

model = ChatOpenAI(temperature=0, model="gpt-4o-2024-05-13", max_tokens=1024)

# RAG pipeline
chain = (
    {
        "context": retriever | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(prompt_func)
    | model
    | StrOutputParser()
)

# Chat interface
if st.session_state.image_details:
    st.write("### Ask a question about the image and PDF")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    prompt = st.chat_input("Ask something about the image and PDF")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        query_input = {"question": prompt}
        response = chain.invoke(query_input)
        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            st.write(response)