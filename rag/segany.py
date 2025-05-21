from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

device = "cuda"
sam = sam_model_registry["vit_h"](checkpoint="/pic/sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
# Load image
image = cv2.imread("/pic/test/r.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate masks
masks = mask_generator.generate(image)
print(len(masks))
print(masks[0].keys())
# Save directory
save_dir = "/pic/segmented_masks"
os.makedirs(save_dir, exist_ok=True)
show_anns(masks)
segmented_images = []

for idx, mask in enumerate(masks):
    segmentation = mask['segmentation']  # Boolean mask
    masked_image = image.copy()
    
    # Apply mask
    masked_image[~segmentation] = 0  # Set background to black
    
    segmented_images.append(masked_image)
    
    # Save each mask image
    out_path = os.path.join(save_dir, f"mask_{idx}.png")
    plt.figure(figsize=(6, 6))
    plt.imshow(masked_image)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

print(f"Saved {len(segmented_images)} segmented images.")

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
#plt.savefig("/pic/seg.png")