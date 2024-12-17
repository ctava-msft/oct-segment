import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import argparse  # Added import for argparse
from predictor import SamPredictor  # Import SamPredictor

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

def split_image_into_layers(image):
    height, width = image.shape[:2]
    layers = [
        image[0:height//2, 0:width//2],        # Top-left quadrant
        image[0:height//2, width//2:width],    # Top-right quadrant
        image[height//2:height, 0:width//2],   # Bottom-left quadrant
        image[height//2:height, width//2:width]  # Bottom-right quadrant
    ]
    return layers

def extract_inputs_from_masks(masks):
    if not masks:
        return None, None, None, None

    point_coords = []
    point_labels = []
    boxes = []
    mask_inputs = []

    for mask in masks:
        # Get the bounding box in [x1, y1, x2, y2] format
        x, y, w, h = mask['bbox']
        box = np.array([x, y, x + w, y + h])

        # Find a point inside the mask
        segmentation = mask['segmentation']
        ys, xs = np.nonzero(segmentation)
        if len(xs) == 0 or len(ys) == 0:
            continue  # Skip if no points found
        idx = np.random.randint(0, len(xs))
        point = np.array([xs[idx], ys[idx]])

        point_coords.append(point)
        point_labels.append(1)  # Label 1 for foreground
        boxes.append(box)
        mask_inputs.append(segmentation)

    if not point_coords:
        return None, None, None, None

    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)
    boxes = np.array(boxes)
    mask_inputs = np.array(mask_inputs)

    return point_coords, point_labels, boxes, mask_inputs

def main(image_path):
    # load image
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    layers = split_image_into_layers(image)  # Define this function
    
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)  # Initialize the predictor
    
    for i, layer in enumerate(layers):
        # Generate masks for the current layer
        masks = mask_generator.generate(layer)
        
        # Convert masks to the required inputs for SamPredictor
        point_coords, point_labels, box, mask_input = extract_inputs_from_masks(masks)
        
        # Set the current layer image in the predictor
        predictor.set_image(layer)
        
        # Predict masks using the extracted inputs
        masks, scores, low_res_masks = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=True
        )
        
        # Process or save the outputs as needed
        # ...

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

