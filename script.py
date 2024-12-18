import cv2
import numpy as np
import os
import torch
from segment_anything import sam_model_registry, SamPredictor
import json
 
# Global variables
steps = 10 # Number of morphing steps
save_counter = 0 # Counter for saved images
 
# Paths to SAM checkpoint and images
SAM_CHECKPOINT = "./sam_vit_b_01ec64.pth"
SAM_CHECKPOINT = "./sam_vit_h_4b8939.pth"
source_image_path = r"./data/samples/image.png"
target_image_path = r"./data/samples/image.png"
 
# Initialize SAM
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.to(device)
predictor = SamPredictor(sam)
 
 
def ensure_segments_folder():
    """Ensure that a 'segments' folder exists."""
    if not os.path.exists("segments"):
        os.makedirs("segments")
 
 
def segment_with_sam(image, image_name="Image"):
    """
    Segment an image using SAM.
    """
    # Convert the image to RGB if it's not already
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4: # Remove alpha channel if present
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    predictor.set_image(image)
    
    # Automatic segmentation mask generation
    masks, scores, _ = predictor.predict_torch(None, None, multimask_output=True)
    return masks, scores
 
 
def save_segments_and_metadata(image, masks, scores, prefix):
    """
    Save each segment as a transparent PNG and store its metadata.
    """
    metadata = []
    ensure_segments_folder()
    
    # Create RGBA version of the image
    rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Create a transparent background
        transparent_mask = np.zeros_like(rgba_image, dtype=np.uint8)
        transparent_mask[:, :, :3] = rgba_image[:, :, :3]
        transparent_mask[:, :, 3] = (mask.cpu().numpy() * 255).astype(np.uint8)
        
        # Save the mask as a PNG
        segment_path = os.path.join("segments", f"{prefix}_segment_{i:04d}.png")
        cv2.imwrite(segment_path, transparent_mask)
    
        # Store metadata
        contours, _ = cv2.findContours((mask.cpu().numpy() * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            metadata.append({
            "segment_id": i,
            "score": float(score.cpu().numpy()),
            "bounding_box": [x, y, x + w, y + h],
            "segment_path": segment_path
            })
    
    return metadata
 
 
def create_mesh_from_segments(metadata):
    """
    Create a morphing mesh using the coordinates of the segments.
    """
    mesh_points = []
    for segment in metadata:
        x_min, y_min, x_max, y_max = segment["bounding_box"]
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        mesh_points.append((center_x, center_y))
    return mesh_points
 
 
def interpolate_points(source_points, target_points, alpha):
    """
    Interpolate between source and target points.
    """
    return [(int((1 - alpha) * x1 + alpha * x2), int((1 - alpha) * y1 + alpha * y2))
    for (x1, y1), (x2, y2) in zip(source_points, target_points)]
 
 
def warp_image_with_mesh(image, grid_points, original_points):
    """
    Warp the image dynamically using the modified mesh grid points.
    """
    h, w = image.shape[:2]
    original_points = np.array(original_points, dtype=np.float32)
    grid_points = np.array(grid_points, dtype=np.float32)
    
    # Delaunay triangulation
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    subdiv.insert([tuple(pt) for pt in original_points])
    triangles = subdiv.getTriangleList().astype(np.int32)
    
    # Create mapping grids
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    
    for t in triangles:
        pts_src = np.array([[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]], dtype=np.float32)
        pts_dst = []
    
        for src_pt in pts_src:
            idx = np.where((original_points == src_pt).all(axis=1))[0]
            if len(idx) > 0:
                pts_dst.append(grid_points[idx[0]])

        pts_dst = np.array(pts_dst, dtype=np.float32)
    
        if len(pts_dst) == 3:
            warp_mat = cv2.getAffineTransform(pts_src, pts_dst)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(pts_src), 255)
    mask_coords = np.where(mask == 255)
    
    for i, j in zip(*mask_coords):
        pt = np.array([j, i, 1], dtype=np.float32)
        dst_pt = np.dot(warp_mat, pt)
        map_x[i, j] = dst_pt[0]
        map_y[i, j] = dst_pt[1]
    
    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped
 
 
def main():
    global save_counter
    
    # Load source and target images
    source_image = cv2.imread(source_image_path)
    target_image = cv2.imread(target_image_path)
    
    if source_image is None or target_image is None:
        print("Error: Could not load one or both images.")
        return
    
    # Ensure both images have the same dimensions
    target_image = cv2.resize(target_image, (source_image.shape[1], source_image.shape[0]))
    
    # Segment source and target images with SAM
    source_masks, source_scores = segment_with_sam(source_image, "Source Image")
    target_masks, target_scores = segment_with_sam(target_image, "Target Image")
    
    # Save segments and metadata
    source_metadata = save_segments_and_metadata(source_image, source_masks, source_scores, prefix="source")
    target_metadata = save_segments_and_metadata(target_image, target_masks, target_scores, prefix="target")
    
    # Save metadata to JSON
    with open("segments/source_metadata.json", "w") as f:
        json.dump(source_metadata, f, indent=4)
    with open("segments/target_metadata.json", "w") as f:
        json.dump(target_metadata, f, indent=4)
    
    # Create mesh grids from segments
    source_points = create_mesh_from_segments(source_metadata)
    target_points = create_mesh_from_segments(target_metadata)
    
    # Morph through steps
    for i in range(steps + 1):
        alpha = i / steps # Interpolation factor
        interpolated_points = interpolate_points(source_points, target_points, alpha)
    
        # Warp source image using interpolated points
        morphed_image = warp_image_with_mesh(source_image, interpolated_points, source_points)
        
        # Blend with target image
        blended_image = cv2.addWeighted(morphed_image, 1 - alpha, target_image, alpha, 0)
        
        # Save the intermediate frame
        save_path = f"new_morphed_step_{save_counter:04d}.jpg"
        cv2.imwrite(save_path, blended_image)
        print(f"Saved: {save_path}")
        save_counter += 1
        
        # Display for visualization
        cv2.imshow("Morphing", blended_image)
        if cv2.waitKey(30) == 27: # ESC key to exit early
            break
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()