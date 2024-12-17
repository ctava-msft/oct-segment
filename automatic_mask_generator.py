import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import argparse  # Added import for argparse

# Show the masks on the image
def show_annontations(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((np.array(sorted_anns[0]['segmentation']).shape[0], np.array(sorted_anns[0]['segmentation']).shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def main(image_path):
    # load image
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    # Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:
    # * `segmentation` : the mask
    # * `area` : the area of the mask in pixels
    # * `bbox` : the boundary box of the mask in XYWH format
    # * `predicted_iou` : the model's own prediction for the quality of the mask
    # * `point_coords` : the sampled input point that generated this mask
    # * `stability_score` : an additional measure of mask quality
    # * `crop_box` : the crop of the image used to generate this mask in XYWH format
    
    masks = mask_generator.generate(image)
    for mask in masks:
        mask['segmentation'] = mask['segmentation'].tolist()
    
    print(len(masks))
    print(masks[0].keys())
    
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'samples', 'masks.json')
    random_suffix = random.randint(1000, 9999)
    output_path = output_path.replace(".json", f"_{random_suffix}.json")
    
    with open(output_path, 'w') as f:
        json.dump(masks, f, indent=4)
    
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_annontations(masks)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Mask Generator")
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()
    main(args.image_path)

