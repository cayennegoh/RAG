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
 
# Suppress warnings
warnings.filterwarnings("ignore")
os.makedirs("Molmo_pts", exist_ok=True)
def read_image_url(image_url):
   response = requests.get(image_url)
   image_pil = Image.open(BytesIO(response.content))
   image_np = np.array(image_pil)
   return image
 
def read_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found or cannot be read from path: {path}")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image

 
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
# MolmoProcessor:
# - image_processor: MolmoImageProcessor {
#     "auto_map": {
#         "AutoImageProcessor": "allenai/MolmoE-1B-0924--image_preprocessing_molmo.MolmoImageProcessor",
#         "AutoProcessor": "allenai/MolmoE-1B-0924--preprocessing_molmo.MolmoProcessor"
#     },
#     "base_image_input_size": [336, 336],
#     "do_normalize": true,
#     "image_mean": [0.48145466, 0.4578275, 0.40821073],
#     "image_std": [0.26862954, 0.26130258, 0.27577711],
#     "image_padding_mask": true,
#     "image_patch_size": 14,
#     "image_processor_type": "MolmoImageProcessor",
#     "image_token_length_h": 12,
#     "image_token_length_w": 12,
#     "max_crops": 12,
#     "overlap_margins": [4, 4],
#     "processor_class": "MolmoProcessor"
# }
def molmo_answer(query_text, input_img):
 
    inputs = processor.process(images=input_img, text=query_text)
    # Move inputs to the correct device and create a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
 
    # Generate output; maximum 2048 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=2048, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer,
    )
 
    # Only get generated tokens; decode them to text
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )
    print(generated_text)
    return generated_text

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
 
    return coordinates
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
 
    return image
def main():
    # Path to image
    img_path = input("in")
    image = read_image(img_path)

    # Prompt to locate objects or points
    query = input("input")
    # Run inference with Molmo
    generated_text = molmo_answer(query, image)
    print("Generated Text:\n", generated_text)

    # Extract coordinates from model output
    coords = get_coords(image, generated_text)
    print("Extracted Coordinates:", coords)

    if coords:
        # Overlay points on the image
        final_image = overlay_points_on_image(image, coords)

        # Display with matplotlib
        
        # plt.axis("off")
        # plt.title("Points Overlay")
        # plt.savefig("/pic/test/output_plot.png")

    else:
        print("No coordinates found in output.")

if __name__ == "__main__":
    main()
