import argparse
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
from scipy.ndimage import gaussian_filter
from scipy import signal
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from predictor import SamPredictor

exp = 1
exp_i = 2
output_dir = f"_output-{exp}-{exp_i}"
os.makedirs(output_dir, exist_ok=True)

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

# Apply a high-pass filter to enhance edges
def filter_signal(image):
    # Updated high-pass filter specifications
    Fs = 1000        # Sample rate (Hz)
    fpass = 200      # Passband cutoff (Hz)
    fstop = 150      # Stopband cutoff (Hz)
    fc = 175         # Highpass cutoff frequency (Hz)
    deltadB = 60     # Minimum desired attenuation in stopband
    beta = 8.6       # Kaiser window beta parameter    

    # Design FIR High-Pass Filter
    M, beta = signal.kaiserord(deltadB, (fpass - fstop) / (Fs / 2))
    if M % 2 == 0:
        M += 1  # Ensure M is odd
    b = signal.firwin(M, fc, window=('kaiser', beta), pass_zero=False, fs=Fs)

    # Apply High-Pass Filter to Image
    return signal.lfilter(b, 1, image)

# Use Canny edge detection to find contours
def draw_contours_from_filtered_image(image):
    # Ensure the image is in uint8 format
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    output_image = image.copy()
    if len(image.shape) == 2:
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Apply Gaussian smoothing
    smoothed_image = gaussian_filter(output_image, sigma=0.9).astype(np.uint8)
    # Use Canny edge detection with adjusted thresholds
    # threshold1=50, threshold2=150
    # threshold1=10, threshold2=70
    t1=10
    t2=70
    edges = cv2.Canny(smoothed_image, threshold1=t1, threshold2=t2)
    # Apply dilation to connect fragmented edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    # Invert edges for correct visualization
    edges = cv2.bitwise_not(edges)
    # Save the edges image for debugging
    cv2.imwrite(f"./{output_dir}/edges_debug.png", edges)
    # Find contours from the edges
    #contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_tuple = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ##print(f"Contours tuple: {contours_tuple}")

    # # Ensure contours_tuple is a tuple and has at least one element
    # if not isinstance(contours_tuple, tuple) or len(contours_tuple) == 0:
    #     raise TypeError(f"Contours tuple should be a non-empty tuple, but got {type(contours_tuple)} with length {len(contours_tuple)}")
    
    contours = contours_tuple[0]  # Extract the contours list
    # If contours is a tuple, convert it to a list
    if isinstance(contours, tuple):
        contours = list(contours)

    MIN_AREA = 1000
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
    print(f"Number of contours detected: {len(contours)}")
    return contours

# Split the image into layers using contours
def split_image_into_layers(image):
    # Apply high-pass filter to enhance edges
    filtered_image = filter_signal(image)
    # Get contours from the filtered image
    contours = draw_contours_from_filtered_image(filtered_image)
    layers = []
    layer_contours = []
    for contour in contours:
        # Create a mask for the contour
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        # Extract the layer using the mask
        layer = cv2.bitwise_and(image, image, mask=mask)
        layers.append(layer)
        layer_contours.append(contour)
    return layers, layer_contours

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
            continue
        idx = np.random.randint(0, len(xs))
        point = np.array([xs[idx], ys[idx]])

        point_coords.append(point)
        point_labels.append(1)
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
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    layers, layer_contours = split_image_into_layers(image)
    
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)
    
    masks = []  # Initialize masks before the loop
    
    for i, (layer, contour) in enumerate(zip(layers, layer_contours)):
        masks.extend(mask_generator.generate(layer))
        predictor.set_image(layer)
        for mask in masks:
            point_coords, point_labels, box, mask_input = extract_input_from_mask(mask)
            if point_coords is None:
                continue
            # Resize mask_input to (1, 256, 256)
            H, W = layer.shape[:2]
            mask_input_resized = cv2.resize(mask_input.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_NEAREST)
            mask_input_resized = mask_input_resized[np.newaxis, np.newaxis, :, :]
            # Predict mask using the extracted inputs
            new_masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                mask_input=mask_input_resized,
                multimask_output=True
            )
        
        # Extract coordinates from contour
        coords = contour.squeeze()
        if len(coords.shape) != 2:
            continue  # Skip if contour is not valid
        # Separate coordinates into top and bottom based on y-value
        coords = coords[np.argsort(coords[:, 0])]  # Sort by x-coordinate
        y_median = np.median(coords[:, 1])
        layer_top = coords[coords[:, 1] <= y_median].tolist()
        layer_bottom = coords[coords[:, 1] > y_median].tolist()
        # Store or print the coordinate lists
        print(f"layer_{i+1}_top = {layer_top}")
        print(f"layer_{i+1}_bottom = {layer_bottom}")

    output_path = os.path.join(os.path.dirname(__file__), 'data', 'samples', 'layer-coordinates.json')
    random_suffix = random.randint(1000, 9999)
    output_path = output_path.replace(".json", f"_{random_suffix}.json")
    
    with open(output_path, 'w') as f:
        json.dump(new_masks, f, indent=4)
    
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_annontations(masks)
    plt.axis('off')
    plt.show()

def extract_input_from_mask(mask):
    # Extract inputs for a single mask
    x, y, w, h = mask['bbox']
    box = np.array([x, y, x + w, y + h])
    segmentation = mask['segmentation']
    ys, xs = np.nonzero(segmentation)
    if len(xs) == 0 or len(ys) == 0:
        return None, None, None, None
    idx = np.random.randint(0, len(xs))
    point_coords = np.array([[xs[idx], ys[idx]]])
    point_labels = np.array([1])
    return point_coords, point_labels, box, segmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Mask Generator")
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()
    main(args.image_path)

