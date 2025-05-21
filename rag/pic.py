import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
def remove_text_from_image(image_path):
    # Load image with OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use pytesseract to get bounding boxes of detected text
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    # Create mask for inpainting
    mask = np.zeros_like(gray)

    for i in range(len(data["text"])):
        if int(data["conf"][i]) > 50:  # Confidence threshold
            (x, y, w, h) = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # Inpaint the masked areas
    inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Convert back to PIL for display or saving
    return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))

# --- Main ---
def main():
    dataset_folder='/pic/dataset'
    image_file = [f for f in os.listdir(dataset_folder)]
    print(image_file)
    for i in image_file: # Path to your image
        input_path = [os.path.join(dataset_folder, f) for f in image_file]
        print(input_path)
        compare_folder='/pic/compare'
        # for  in image_file:
        
        output_path = os.path.join(compare_folder, i)
        print(output_path)
            # Remove text and show result
        
        clean_img = remove_text_from_image(os.path.join(dataset_folder,i))
        clean_img.show()
            
                # Save result as a new image
        cleaned_np = cv2.cvtColor(np.array(clean_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, cleaned_np)

        print(f"Cleaned image saved to: {output_path}")
