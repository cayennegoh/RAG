from matplotlib import pyplot as plt
import os
from PIL import Image
import warnings
from dotenv import load_dotenv
import base64
import io
import numpy as np


path = "/pic/"


# Extract images, tables, and chunk text
from unstructured.partition.pdf import partition_pdf

raw_pdf_elements = partition_pdf(
    filename=path + "photo1.pdf",
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

#save_images(ds, dataset_folder, num_images=1500)  # Uncomment to save images

dataset_folder="/pic"

# == Setup the chromaDB ==
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

# create a chromadb object
chroma_client = chromadb.PersistentClient(path="./pic.db")

# instantiate image loader
image_loader = ImageLoader()

# instantiate multimodal embedding function
embedding_function = OpenCLIPEmbeddingFunction()

# create the collection, - vector database
vectorstore = chroma_client.get_or_create_collection(
    "fashion_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)
vectorstore.add(texts)
ids = []
uris = []

### Iterate over each file in the dataset folder -- uncomment this for the first time run ###
for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        file_path = os.path.join(dataset_folder, filename)

        # Append id and uri to respective lists
        ids.append(str(i))
        uris.append(file_path)

# # Assuming multimodal_db is already defined and available
fashion_collection.add(ids=ids, uris=uris)  # Uncomment to add images to the database

print("Images added to the database.")

# # Validate the VectorDB with .count()
print(fashion_collection.count())  # res: 500


# === Functions for Querying the VectorDB ===
def query_db(query, results=3):
    print(f"Querying the database for: {query}")
    results = fashion_collection.query(
        query_texts=[query], n_results=results, include=["uris", "distances"]
    )
    return results


def print_results(results):
    for idx, uri in enumerate(results["uris"][0]):
        print(f"ID: {results['ids'][0][idx]}")
        print(f"Distance: {results['distances'][0][idx]}")
        print(f"Path: {uri}")
        # Display the image using matplotlib
        show_image_from_uri(uri)
        print("\n")


# query = "black pants"  # Change the query to test different images or different
# results = query_db(query)
# print_results(results)


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import base64

# Instantiate the OpenAI model
vision_model = ChatOpenAI(
    model="gpt-4o", temperature=0.0
)  # this model has vision capabilities

# instantiate the output parser
parser = StrOutputParser()


# Define the prompt template
image_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a talented fashion designer and you have been asked to style outfit for a special event. Answer the user's question  using the given image context with direct references to parts of the images provided."
            " Maintain a more conversational tone, don't make too many lists. Use markdown formatting for highlights, emphasis, and structure.",
        ),
        (
            "user",
            [
                {
                    "type": "text",
                    "text": "what are some good ideas to match the outfit {user_query}",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_data_1}",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_data_2}",
                },
            ],
        ),
    ]
)

# Define the LangChain Chain
vision_chain = image_prompt | vision_model | parser


# === Foramtting query results for LLM prompting ===
# to input the images in as context, we need to first encode the images as base64 strings for the LLM to understand


# The function below that will do that, and create a dictionary along with
# the original user query to pass into the chain. The chain will take a dictionary input,
# that will correspond to the three pieces of information
# that need to be injected into it {user_query}, {image_data_1}, {image_data_2}.
def format_prompt_inputs(data, user_query):
    print("Formatting prompt inputs...")
    inputs = {}

    # Add user query to the dictionary
    inputs["user_query"] = user_query

    # Get the first two image paths from the 'uris' list
    image_path_1 = data["uris"][0][0]
    image_path_2 = data["uris"][0][1]

    # Encode the first image
    with open(image_path_1, "rb") as image_file:
        image_data_1 = image_file.read()
    inputs["image_data_1"] = base64.b64encode(image_data_1).decode("utf-8")

    # Encode the second image
    with open(image_path_2, "rb") as image_file:
        image_data_2 = image_file.read()
    inputs["image_data_2"] = base64.b64encode(image_data_2).decode("utf-8")

    # inputs dictionary will have the user query and the base64 encoded images and will look like this:
    # {
    #     "user_query": "pink flower with yellow center",
    #     "image_data_1": "base64_encoded_image_1",
    #     "image_data_2": "base64_encoded_image_2"
    # }
    print("Prompt inputs formatted....")
    return inputs


def image_to_image_search(image_path, results=3):
    print(f"Searching for similar images to: {image_path}")
    
    try:
        # Open and process the image
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = img.resize((224, 224))  # Resize to match CLIP's expected input size
            img_array = np.array(img)
        
        # Generate embedding using the OpenCLIPEmbeddingFunction
        embedding = embedding_function([img_array])  # This should return a list with one embedding
        
        if not embedding or len(embedding) == 0:
            raise ValueError("Embedding generation failed")
        
        # Perform the search using the generated embedding
        results = fashion_collection.query(
            query_embeddings=embedding,
            n_results=results,
            include=["uris", "distances"]
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

def main():
    print("Welcome to the Fashion Styling service!")
    print("1. Text-to-Image Search")
    print("2. Image-to-Image Search")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # Existing text-to-image search code
        pass
    elif choice == "2":
        image_path = input("Enter the path to your query image: \n")
        results = image_to_image_search(image_path, results=3)
        
        if results and results["uris"] and len(results["uris"][0]) > 0:
            print("\nHere are similar products based on your image query:\n")
            for uri in results["uris"][0]:
                show_image_from_uri(uri)
                print(f"Image URI: {uri}")
                print()
        else:
            print("Unable to find similar products or process the image.")
    else:
        print("Invalid choice. Please run the program again and select 1 or 2.")

if __name__ == "__main__":
    main()