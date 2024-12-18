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
from segment_anything import sam_model_registry
from segment_anything.predictor import SamPredictor
from segment_anything import SamAutomaticMaskGenerator

exp = 1
exp_i = 5
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


def draw_contours_from_filtered_image(image):

    # Ensure the image is in uint8 format
    if image.dtype != np.uint8:
        bscan = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    output_image = image.copy()
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
    #print(f"Type of contours: {type(contours)}")
    #print(f"Sample contour: {contours[0] if len(contours) > 0 else 'No contours'}")
    # Ensure contours is a list of numpy arrays
    if not isinstance(contours, list):
        raise TypeError(f"Contours should be a list, but got {type(contours)}")
    for i, contour in enumerate(contours):
        if not isinstance(contour, np.ndarray):
            raise TypeError(f"Contour at index {i} should be a numpy array, but got {type(contour)}")
    # Define layer colors based on the legend (approximation)
    layer_colors = [
        (255, 0, 0),    # Blue for RNFL
        (255, 165, 0),  # Orange for GCL
        (0, 255, 0),    # Green for IPL
        (255, 0, 255),  # Purple for INL
        (255, 255, 0),  # Yellow for OPL
        (255, 69, 0),   # Red-Orange for ONL/ELM
        (255, 20, 147), # Pink for EZ
        (255, 0, 0),    # Red for POS
        (0, 255, 255),  # Cyan for RPE/BM
    ]
    layer_coordinates = {}
    # print(f"Layer: {layer_idx}")
    # # Handle layer_colors as a list
    # color = layer_colors[layer_idx+1] if layer_idx < len(layer_colors) else (255, 255, 255)
    # cv2.drawContours(output_image, contours, -1, color, 2)        
    # #line_coordinates[layer_idx] = contours.tolist()
    # # Print contour information
    # coords = contours[layer_idx].squeeze()
    # coords = coords[np.argsort(coords[:, 0])]
    # y_median = np.median(coords[:, 1])
    # layer_top = coords[coords[:, 1] <= y_median].tolist()
    # layer_bottom = coords[coords[:, 1] > y_median].tolist()
    # layer_coordinates[f'layer_{layer_idx+1}_top'] = layer_top
    # layer_coordinates[f'layer_{layer_idx+1}_bottom'] = layer_bottom

    # Generate input_points, input_boxes, and input_labels based on contours
    input_points = []
    input_boxes = []
    input_labels = []
    
    for idx, contour in enumerate(contours):
        # Calculate centroid for input_points
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            input_points.append([cX, cY])
        
        # Calculate bounding box for input_boxes
        x, y, w, h = cv2.boundingRect(contour)
        input_boxes.append([x, y, x + w, y + h])
        
        # Assign label based on layer_config or default value
        label = f"Layer_{idx}"
        input_labels.append(label)
    
    # Print or save the lists as needed
    print("input_points:", input_points)
    print("input_boxes:", input_boxes)
    print("input_labels:", input_labels)
    
    return output_image, layer_coordinates

# Use Canny edge detection to find contours
def detect_contours_from_filtered_image(image):
    # Ensure the image is in uint8 format
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    output_image = image.copy()
    if len(image.shape) == 2:
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Enhance contrast using CLAHE
    if len(image.shape) == 2:
        image_gray = image
    else:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_equalized = clahe.apply(image_gray)
    # Apply Gaussian smoothing with adjusted sigma
    smoothed_image = gaussian_filter(image_equalized, sigma=1).astype(np.uint8)
    # Adjust thresholds for Canny edge detection
    t1 = 30
    t2 = 90
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

    contours = contours_tuple[0]  # Extract the contours list
    # If contours is a tuple, convert it to a list
    if isinstance(contours, tuple):
        contours = list(contours)

    MIN_AREA = 1000
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
    print(f"Number of contours detected: {len(contours)}")

    # Extract top and bottom coordinates for each contour
    layer_coordinates = {}
    for idx, contour in enumerate(contours):
        coords = contour.squeeze()
        if len(coords.shape) != 2:
            continue  # Skip invalid contours
        y_median = np.median(coords[:, 1])
        layer_top = coords[coords[:, 1] <= y_median].tolist()
        layer_bottom = coords[coords[:, 1] > y_median].tolist()
        layer_coordinates[f'layer_{idx+1}_top'] = layer_top
        layer_coordinates[f'layer_{idx+1}_bottom'] = layer_bottom

    # Save the layer coordinates to a JSON file
    json_path = os.path.join(output_dir, 'layer_coordinates.json')
    with open(json_path, 'w') as f:
        json.dump(layer_coordinates, f, indent=4)

    return contours, layer_coordinates

def detect_line_segments(image, edge_threshold1=50, edge_threshold2=150, hough_threshold=50, min_line_len=20, max_line_gap=10):
    """
    Detect line segments using edge detection and probabilistic Hough transform.
    You might need to tweak parameters (edge_threshold1, edge_threshold2, hough_threshold, etc.)
    to get meaningful results depending on your images.
    """
    # Detect edges using Canny
    edges = cv2.Canny(image, edge_threshold1, edge_threshold2)
 
    # Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines, edges

# Split the image into layers using contours
def split_image_into_layers(image):
    # Apply high-pass filter to enhance edges
    filtered_image = filter_signal(image)
    # # Detect contours from the filtered image
    # contours = detect_contours_from_filtered_image(filtered_image)

    lines, edges = detect_line_segments(image)
    print(lines)
    print(edges)

    draw_lines_and_edges(image,lines,edges)
    
    # Write lines and edges to JSON files
    with open(os.path.join(output_dir, 'lines.json'), 'w') as f:
        json.dump(lines.tolist(), f)
    with open(os.path.join(output_dir, 'edges.json'), 'w') as f:
        json.dump(edges.tolist(), f)

    # Draw contours from the filtered image
    contours, layer_coordinates = detect_contours_from_filtered_image(filtered_image)
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

layer_colors = [
    (255, 0, 0),    # Blue for RNFL
    (255, 165, 0),  # Orange for GCL
    (0, 255, 0),    # Green for IPL
    (255, 0, 255),  # Purple for INL
    (255, 255, 0),  # Yellow for OPL
    (255, 69, 0),   # Red-Orange for ONL/ELM
    (255, 20, 147), # Pink for EZ
    (255, 0, 0),    # Red for POS
    (0, 255, 255),  # Cyan for RPE/BM
]

def draw_lines_and_edges(image, lines, edges):
    # Draw lines with corresponding layer colors
    for idx, line in enumerate(lines):
        if len(line) < 4:
            continue  # Skip lines that do not have enough points
        color = layer_colors[idx % len(layer_colors)]
        cv2.line(image, (line[0], line[1]), (line[2], line[3]), color, 2)
    
    # Convert edges to BGR and overlay on the image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined_image = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
    
    # Save the combined image to the output directory
    output_path = os.path.join(output_dir, 'lines_edges.png')
    cv2.imwrite(output_path, combined_image)

def main(image_path):
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    layers, layer_contours = split_image_into_layers(image)
    print(f"Number of layers: {len(layers)}")
    print(f"Number of layer contours: {len(layer_contours)}")
    
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
    # parser = argparse.ArgumentParser(description="Automatic Mask Generator")
    # parser.add_argument('image_path', type=str, help='Path to the input image')
    # args = parser.parse_args()
    image_path="./data/samples/image.png"
    main(image_path)

