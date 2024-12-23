import argparse
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import json
import torch.nn as nn
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

exp = 1
exp_i = 1
output_dir = f"_output-{exp}-{exp_i}"
os.makedirs(output_dir, exist_ok=True)

class CannyFilter(nn.Module):

    def get_gaussian_kernel(self, k=3, mu=0, sigma=1, normalize=True):
        # compute 1 dimension gaussian
        gaussian_1D = np.linspace(-1, 1, k)
        # compute a grid distance from center
        x, y = np.meshgrid(gaussian_1D, gaussian_1D)
        distance = (x ** 2 + y ** 2) ** 0.5

        # compute the 2 dimension gaussian
        gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
        gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

        # normalize part (mathematically)
        if normalize:
            gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
        return gaussian_2D

    def get_sobel_kernel(self, k=3):
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        return sobel_2D

    def get_thin_kernels(self):
        import numpy as np
        kernels = []
        for _ in range(8):
            kernel = np.zeros((3, 3), dtype=np.float32)
            # Simple placeholder for thinning kernels
            kernel[1, 1] = 1.
            kernels.append(kernel)
        return kernels

    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=False):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'

        # gaussian

        gaussian_2D = self.get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        with torch.no_grad():
            self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)

        # sobel

        sobel_2D = self.get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        
        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=k_sobel,
                                    padding=k_sobel // 2,
                                    bias=False)
        with torch.no_grad():
            self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)
            self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)


        # thin

        thin_kernels = self.get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        with torch.no_grad():
            self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        with torch.no_grad():
            self.hysteresis.weight[:] = torch.from_numpy(hysteresis)


    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the steps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        for c in range(C):
            # apply gaussian
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])

            # apply sobel filter
            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])

        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges

        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1


        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges

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

def create_image_from_coords(coords, output_dir):
    import numpy as np
    import cv2
    height, width, _ = coords.shape
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw lines based on gradient coordinates
    scaling_factor = 10  # Scale gradients for better visibility
    for y in range(height):
        for x in range(width):
            dx, dy = coords[y, x]
            if not np.allclose([dx, dy], [0, 0]):
                end_x = int(x + dx * scaling_factor)
                end_y = int(y + dy * scaling_factor)
                # Ensure end coordinates are within image bounds
                end_x = max(0, min(end_x, width - 1))
                end_y = max(0, min(end_y, height - 1))
                color = layer_colors[y % len(layer_colors)]
                cv2.line(blank_image, (x, y), (end_x, end_y), color, 1)
    output_path = os.path.join(output_dir, "new_image_from_coords.png")
    cv2.imwrite(output_path, blank_image)

# draws the thin edge contours
def draw_thin_contours(thin_edges, output_dir, grad_x=None, grad_y=None, grad_magnitude=None, grad_orientation=None):
    thin_edges_np = thin_edges.squeeze().detach().cpu().numpy()
    thin_edges_np = (thin_edges_np * 255).astype(np.uint8)  # Scale to 0-255
    img = np.zeros((thin_edges_np.shape[0], thin_edges_np.shape[1], 3), dtype=np.uint8)
    # Convert orientation to NumPy if provided
    if grad_orientation is not None:
        grad_orientation_np = grad_orientation.squeeze().detach().cpu().numpy()
    else:
        grad_orientation_np = None

    for y in range(thin_edges_np.shape[0]):
        for x in range(thin_edges_np.shape[1]):
            if thin_edges_np[y, x]:
                if grad_orientation_np is not None:
                    orientation_value = grad_orientation_np[y, x]
                    orientation_index = int((orientation_value // 45) % len(layer_colors))
                    color = layer_colors[orientation_index]
                else:
                    color = layer_colors[y % len(layer_colors)]
                img[y, x] = color

    output_path = os.path.join(output_dir, "thin_contours.png")
    cv2.imwrite(output_path, img)

def load_thin_edges(file_path):
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return np.array(json.load(f))
    else:
        raise ValueError("Unsupported file format. Use .npy or .json")

def create_new_image_from_thin_edges(thin_edges, output_dir):
    # Squeeze to remove batch and channel dimensions
    img = (thin_edges * 255).squeeze().detach().cpu().numpy().astype(np.uint8)
    image = Image.fromarray(img)
    image.save(os.path.join(output_dir, 'thin_edges_new_image.png'))

# Main method
def main(image_path):
    # Edge Detection
    image = Image.open(image_path).convert('RGB')
    image.save(os.path.join(output_dir, os.path.basename(image_path)))
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    canny = CannyFilter(k_gaussian=3, mu=0, sigma=1, k_sobel=3, use_cuda=False)
    blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges = canny(image_tensor)
    vutils.save_image(thin_edges, os.path.join(output_dir, "thin_edges.png"))


    # Draw contours on the image
    draw_thin_contours(thin_edges, output_dir, grad_x, grad_y, grad_magnitude, grad_orientation)
    
    # Create new image from thin_edges
    create_new_image_from_thin_edges(thin_edges, output_dir)

    # Save thin edges as numpy array and JSON
    thin_edges_np = thin_edges.squeeze().detach().cpu().numpy()
    np.save(os.path.join(output_dir, "thin_edges.npy"), thin_edges_np)
    with open(os.path.join(output_dir, "thin_edges.json"), 'w') as f:
        json.dump(thin_edges_np.tolist(), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Edge and Mask Generator")
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()
    main(args.image_path)