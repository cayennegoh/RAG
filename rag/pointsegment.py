import os
import logging
from datetime import datetime
import warnings

import cv2
import requests
import re
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    BitsAndBytesConfig,
)

# For SAM integration (install segment_anything library: pip install segment_anything)
import torch
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: segment_anything library not found. SAM integration will be disabled.")

warnings.filterwarnings("ignore")
os.makedirs("Molmo_pts", exist_ok=True)

def read_image_url(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        image_pil = Image.open(BytesIO(response.content)).convert("RGB")
        image_np = np.array(image_pil)
        return image_np
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from URL: {e}")
        return None
    except Exception as e:
        print(f"Error opening image from URL: {e}")
        return None

def read_image(path):
    try:
        image = cv2.imread(path)
        if image is None:
            print(f"Error: Could not read image at path: {path}")
            return None
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image
    except FileNotFoundError:
        print(f"Error: Image not found at path: {path}")
        return None
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

path = "/pic/test/1.png"
img_path = os.path.join(os.getcwd(), path)

model_id = "allenai/MolmoE-1B-0924" #
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=False
)

def load_model(
    model_id, quant_config: BitsAndBytesConfig = None, dtype="auto", device="cuda"
):
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

model, processor = load_model(
    model_id, quant_config=quant_config, dtype="auto", device="cuda"
)

# Load SAM model if available
sam_predictor = None
if SAM_AVAILABLE:
    sam_checkpoint = "path/to/your/sam_vit_h_40b893.pth"  # Replace with your SAM checkpoint path
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        sam_predictor = SamPredictor(sam)
    except Exception as e:
        SAM_AVAILABLE = False
        print(f"Error loading SAM model: {e}")
        print("SAM integration disabled.")

def get_centroid_from_mask(mask):
    y_indices, x_indices = np.where(mask)
    if y_indices.size > 0 and x_indices.size > 0:
        center_x = np.mean(x_indices)
        center_y = np.mean(y_indices)
        return center_x, center_y
    return None

def custom_invoke_with_image(model, processor, user_query=None, context=None, image_path=None, retrieved_path=None, max_length=150, temperature=0.7):
    try:
        img = read_image(image_path)
        if img is None:
            return "No image"
        h, w, _ = img.shape
        sam_point_output = None

        if SAM_AVAILABLE and "turtle" in user_query.lower():  # Example: Use SAM for "turtle" queries
            try:
                predictor.set_image(img)
                # You might need initial points or a more advanced prompting strategy for SAM
                # Here's a very basic example - you'd likely need more sophisticated prompting
                input_point = np.array([[w//2, h//2]]) # Example: center of the image
                input_label = np.array([1])
                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )

                best_mask = None
                # Logic to choose the mask corresponding to the turtle (this is the complex part)
                # You might need to analyze the masks based on size, shape, or other features
                if masks is not None and len(masks) > 0:
                    best_mask = masks[0] # Placeholder: Replace with actual logic

                if best_mask is not None:
                    centroid = get_centroid_from_mask(best_mask)
                    if centroid:
                        x_percent = (centroid[0] / w) * 100
                        y_percent = (centroid[1] / h) * 100
                        sam_point_output = f"<point x=\"{x_percent:.1f}\" y=\"{y_percent:.1f}\" alt=\"SAM identified turtle\" />"
            except Exception as e:
                print(f"Error during SAM inference: {e}")

        if sam_point_output:
            return sam_point_output
        elif any(word.lower() in context.lower() for word in user_query.split()):
            prompt = f" Annotate the point in the query image '{image_path}' that corresponds most similar to the retrieved image '{retrieved_path}'."
        else:
            prompt = f'{user_query}'

        input_text = prompt
        inputs = processor.process(text=input_text, images=[img])
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=max_length, temperature=temperature, stop_strings=["<|endoftext|>" ]),
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

def get_coords(image, generated_text):
    h, w, _ = image.shape
    coordinates = []

    if "</point" in generated_text:
        matches = re.findall(
            r'(?:x(?:\d*)="([\d.]+)"\s*y(?:\d*)="([\d.]+)")', generated_text
        )
        if matches:
            coordinates = [
                (int(float(x_val) / 100 * w), int(float(y_val) / 100 * h))
                for x_val, y_val in matches
            ]
    else:
        print("There are no points obtained from regex pattern")

    return coordinates

def overlay_points_on_image(image, points, radius=5, color=(255, 0, 0)):
    if image is None:
        return None
    pink_color = (158, 89, 243)  # Color for the points (BGR format)
    annotated_image = image.copy()
    for (x, y) in points:
        # Draw an outline for the point
        cv2.circle(
            annotated_image,
            (int(x), int(y)),
            radius=radius + 1,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        # Draw the point itself
        cv2.circle(
            annotated_image,
            (int(x), int(y)),
            radius=radius,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA
        )

    # Save and convert the image
    sav_image = annotated_image.copy()
    image_rgb = cv2.cvtColor(sav_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("/pic/test/output_pt.jpg", sav_image)

    return image_rgb

def main():
    # Path to image
    img_path = input("Enter image path or URL: ")
    if img_path.startswith("http://") or img_path.startswith("https://"):
        image = read_image_url(img_path)
    else:
        image = read_image(img_path)

    if image is None:
        print("false")
        return

    # Prompt to locate objects or points
    query = input("Enter query: ")
    # Run inference with Molmo
    generated_text = molmo_answer(query, image)
    print("Generated Text:\n", generated_text)

    if generated_text == "no pic" or generated_text == "error":
        print("false")
        return

    # Extract coordinates from model output
    coords = get_coords(image, generated_text)
    print("Extracted Coordinates:", coords)

    if coords:
        # Overlay points on the image
        final_image = overlay_points_on_image(image.copy(), coords) # Use a copy

        if final_image is not None:
            # Display with matplotlib
            plt.imshow(final_image)
            plt.axis("off")
            plt.title("Points Overlay")
            plt.savefig("/pic/test/output_plot.png")
            plt.show()
        else:
            print("false")
    else:
        print("false")

if __name__ == "__main__":
    main()