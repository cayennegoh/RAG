from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
embedding_function=OpenCLIPEmbeddingFunction()
# --- Config ---
query_image_path = "/pic/test/r.png"
reference_image_path = "/pic/dataset/1.png"
save_best_path = "/pic/segmented_masks/best_match.png"
device = "cuda"

# --- Setup SAM ---
sam = sam_model_registry["vit_h"](checkpoint="/pic/sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# --- Load & Segment Query Image ---
image = cv2.imread(query_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)

# --- Embed the Reference Image ---
ref_image = Image.open(reference_image_path).convert("RGB")
ref_image = ref_image.resize((224, 224))  # Resizing image to 224x224 for OpenCLIP model
ref_image = np.array(ref_image)
ref_embedding = embedding_function([ref_image])[0]

# --- Loop through each mask and embed masked region ---
best_score = float("inf")
best_region = None

for idx, mask in enumerate(masks):
    segmentation = mask['segmentation']
    masked_image = image.copy()
    masked_image[~segmentation] = 0  # apply mask

    pil_masked = Image.fromarray(masked_image)
    pil_masked = pil_masked.resize((224, 224))  # Resizing image to 224x224 for OpenCLIP model
    pil_masked = np.array(pil_masked)
    region_embedding = embedding_function([pil_masked])[0]

    # Compute cosine distance
    sim = np.dot(ref_embedding, region_embedding) / (
        np.linalg.norm(ref_embedding) * np.linalg.norm(region_embedding)
    )
    distance = 1 - sim

    if distance < best_score:
        best_score = distance
        best_region = pil_masked.copy()
        best_index=idx

# --- Save & show best match ---
if best_region is not None:
    Image.fromarray(best_region).save(save_best_path)

    print(f"Best matching region saved at: {save_best_path}")
    plt.imshow(best_region)
    plt.axis("off")
    plt.title(f"Most similar region (cosine distance: {best_score:.4f})")
    plt.savefig("/pic/segmented.png")
else:
    print("No matching region found.")
import matplotlib.patches as patches

# Get bounding box of the best matching region
best_bbox = masks[best_index]["bbox"]  # [x0, y0, width, height]
x0, y0, w, h = best_bbox
center_x = x0 + w // 2
center_y = y0 + h // 2

# Draw the original image with a red point at the center
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)
ax.plot(center_x, center_y, 'ro', markersize=15)  # red point
ax.axis('off')

# Save the image with the point
point_path = "/pic/segmented_masks/original_with_point.png"
plt.savefig(point_path, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"Point plotted on original image and saved to: {point_path}")
