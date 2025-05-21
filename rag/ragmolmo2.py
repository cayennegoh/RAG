import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import os
import base64
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

# Processor and Model Loading
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    use_fast=False,
    torch_dtype='auto',
    device_map="cpu"
)

model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    low_cpu_mem_usage=True,
    device_map="cpu"
)

# Define the function to handle inputs
def process_inputs(docs):
    images = []
    text = []
    for doc in docs:
        if isinstance(doc, dict) and "image" in doc:
            # Process image (resize, base64, etc.)
            images.append(doc["image"])  # Assuming doc["image"] is base64 or URL
        elif isinstance(doc, str):
            text.append(doc)
    return {"images": images, "texts": text}
# def prompt_func(data_dict):
#     formatted_texts = "\n".join(data_dict["context"]["texts"])
#     messages = []

#     # Process images through the processor if present
#     if data_dict["context"]["images"]:
#         processed_images = [
#             processor(images=[base64.b64decode(img)], return_tensors="pt")
#             for img in data_dict["context"]["images"]
#         ]
#         processed_images_tensors = [
#             torch.tensor(image['pixel_values']) for image in processed_images
#         ]
#         print({processed_images_tensors})
#     else:
#         processed_images_tensors = None

#     # Format text input properly
#     text_prompt = (
#         "As an expert art critic and historian, analyze the following image(s) "
#         "and text retrieved from a vector database:\n\n"
#         f"User-provided keywords: {data_dict['question']}\n\n"
#         "Retrieved Text:\n"
#         f"{formatted_texts}"
#     )

#     # Process text input through the processor
#     processed_text = processor.tokenizer(text=text_prompt, return_tensors="pt")["input_ids"]
#     text_tensor=torch.tensor(processed_text)
#     return {"text_inputs": text_tensor, "image_inputs": processed_images_tensors}
def prompt_func(data_dict):
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    images = []

    # Process images if they exist in the input
    if data_dict["context"]["images"]:
        images = [
            Image.open(requests.get(img, stream=True).raw) if isinstance(img, str) else img for img in data_dict["context"]["images"]
        ]
    else:
        images = None

    # Combine the text prompt
    text_prompt = (
        "As an expert art critic and historian, analyze the following image(s) "
        "and text retrieved from a vector database:\n\n"
        f"User-provided keywords: {data_dict['question']}\n\n"
        "Retrieved Text:\n"
        f"{formatted_texts}"
    )

    # Now process the inputs with the processor
    inputs = processor.process(
        images=images,
        text=text_prompt
    )
    # Process inputs (ensuring proper tensor format)
    text_tensor = inputs['input_ids'].to(model.device).unsqueeze(0)
    image_tensor = inputs['pixel_values'].to(model.device).unsqueeze(0) if 'pixel_values' in inputs else None

# Now pass these tensors to the model


    
    # # Move inputs to the model's device
    # inputs = processor.process(images=images, text=text_prompt)
    # inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # return inputs

# Create a Chroma vector store and add images (assuming they are in the 'path' variable)
path = "/pic/"
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

# Chain definition
chain = (
    {
        "context": retriever | RunnableLambda(process_inputs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(prompt_func)
    | model
    | StrOutputParser()
)
question=input('q')
query_result=chain.invoke({"question": question})
outputs=model.generate(
    input_ids=text_tensor,
    pixel_values=image_tensor,
    max_new_tokens=500,
    stop_strings=["<|endoftext|>"]
)

# Decode the generated output
generated_tokens = output[0, text_tensor.size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(generated_text)