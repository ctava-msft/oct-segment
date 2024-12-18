import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy import fftpack
from scipy.ndimage import gaussian_filter
from scipy import signal
from scipy.fft import fft, fftfreq

exp = 22
exp_i = 1
output_dir = f"_output-{exp}-{exp_i}"
os.makedirs(output_dir, exist_ok=True)
segment_dir = f"_output-{exp}-{exp_i}/segments"
os.makedirs(segment_dir, exist_ok=True)

# Use High Pass Filter to remove the noise
def filter_signal(bscan):
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

    # Apply High-Pass Filter to bscan Image
    return signal.lfilter(b, 1, bscan)


def draw_contours_from_filtered_bscan(bscan,layers_config):

    # Ensure the image is in uint8 format
    if bscan.dtype != np.uint8:
        bscan = cv2.normalize(bscan, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    output_image = bscan.copy()
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
    for layer_idx in layers_config.keys():
        if layer_idx >= len(contours):
            continue
        print(f"Layer: {layer_idx}")
        # Handle layer_colors as a list
        color = layer_colors[layer_idx+1] if layer_idx < len(layer_colors) else (255, 255, 255)
        cv2.drawContours(output_image, contours, -1, color, 2)        
        #line_coordinates[layer_idx] = contours.tolist()
        # Print contour information

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
        label = layers_config.get(idx, "Layer_{}".format(idx))
        input_labels.append(label)
    
    # Print or save the lists as needed
    print("input_points:", input_points)
    print("input_boxes:", input_boxes)
    print("input_labels:", input_labels)
    
    return output_image

if __name__ == "__main__":
    # Define the path to the image
    image_name = 'image.png'
    #image_name = 'oct-id-105.jpg'
    #image_name = 'kaggle-NORMAL-3099713-1.jpg'
    #image_name = 'oct-500-3-10301-1.bmp'
    image_path = os.path.join(os.path.dirname(__file__), 'images', 'samples', image_name)

    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    cv2.imwrite(f"./{output_dir}/orig-image.png", image)

    # Call signal_finder function
    filtered_bscan = filter_signal(image)

    layers_config = {
        0: 0.25,  # RNFL
        # Add more layers and multipliers as needed
    }

    processed_bscan = draw_contours_from_filtered_bscan(filtered_bscan,layers_config)

    print(f"./{output_dir}/layers-drawn.png")
    cv2.imwrite(f"./{output_dir}/layers-drawn.png", processed_bscan)
    
    # # Define input points, boxes, and labels
    # input_points = [[298, 444], [137, 172], [537, 164], [322, 230], [340, 132], [295, 86], [325, 94]]
    # input_boxes = [[0, 219, 620, 594], [35, 141, 231, 207], [493, 139, 590, 186], [0, 124, 620, 323], [321, 104, 364, 163], [255, 30, 330, 131], [0, 0, 620, 244]]
    # input_labels = [0.25, 'Layer_1', 'Layer_2', 'Layer_3', 'Layer_4', 'Layer_5', 'Layer_6']
    
    # Iterate over labels, points, and boxes to create and save cropped images
    for label, box in zip(input_labels, input_boxes):
        x1, y1, x2, y2 = box
        cropped_image = image[y1:y2, x1:x2]
        output_path = f"./{output_dir}/cropped_{label}.png"
        cv2.imwrite(output_path, cropped_image)