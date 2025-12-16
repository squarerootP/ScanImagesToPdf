import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO  # type: ignore

@dataclass 
class ProcessingConfig:
    model_path: str = "best (4).pt"
    target_width: int = 1000
    target_height: int = 1400
    visualize: bool = False
    allowed_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
    detection_confidence: float = 0.5
    padding_percent: float = 0.03  # 3% padding around detected object mask
    margin_ratio: float = 0.94  # 94% fill ratio when zooming out
 

config = ProcessingConfig()
model = YOLO(config.model_path)
# sam_model = SAM("")

def run_segment(img_path, conf: float = 0.5):
  annotations = model.predict(img_path, conf=conf)
  return annotations

def clean_mask(mask):
    # mask is a binary numpy array (0/1)
    mask_uint8 = (mask * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)
    
    if num_labels <= 1:  # Only background exists
        return mask.astype(np.uint8) 
    
    # stats[:, cv2.CC_STAT_AREA] gives area of each component
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    largest_idx = 1 + np.argmax(areas)   # +1 because background is label 0

    # Create clean mask with only the largest component
    clean = (labels == largest_idx).astype(np.uint8)

    return clean


def expand_mask(mask, expand_percent=0.03):
    """
    Expand the mask by a percentage of its bounding box dimensions.
    
    Args:
        mask: Binary mask (0/1 or 0/255)
        expand_percent: Percentage to expand (0.03 = 3%)
    
    Returns:
        Expanded binary mask
    """
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    
    # Find bounding box to calculate kernel size based on mask dimensions
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # Calculate kernel size as percentage of bounding box dimensions
    kernel_w = max(3, int(w * expand_percent))
    kernel_h = max(3, int(h * expand_percent))
    
    # Make kernel size odd for proper centering
    kernel_w = kernel_w if kernel_w % 2 == 1 else kernel_w + 1
    kernel_h = kernel_h if kernel_h % 2 == 1 else kernel_h + 1
    
    # Create elliptical kernel and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))
    expanded = cv2.dilate(mask_uint8, kernel, iterations=1)
    
    return (expanded / 255).astype(np.uint8) if mask.max() <= 1 else expanded


def zoom_out_to_fit(cropped_img, target_w, target_h, fill_ratio=0.94):
    """
    Place the cropped image onto a canvas of size (target_w, target_h) by zooming out
    (scaling down) until the image width or height reaches fill_ratio (90%) of the 
    target dimension. The image is centered on a white canvas.
    
    Args:
        cropped_img: The cropped upright image (BGR format)
        target_w: Target canvas width
        target_h: Target canvas height
        fill_ratio: The ratio of target dimension that the image should fill (default 0.9 = 90%)
    
    Returns:
        Canvas image with the cropped image centered and scaled appropriately
    """
    crop_h, crop_w = cropped_img.shape[:2]
    
    # Calculate scale factors to make width or height hit fill_ratio of target
    scale_w = (target_w * fill_ratio) / crop_w
    scale_h = (target_h * fill_ratio) / crop_h
    
    # Use the smaller scale to ensure both dimensions fit within the target
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions after scaling
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    
    # Resize the cropped image
    scaled_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Create white canvas of target size
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    
    # Calculate position to center the scaled image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Place the scaled image on the canvas
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = scaled_img
    
    return canvas

def plot_segments(annotations, target_w=1000, target_h=1400, visualize=ProcessingConfig.visualize):
    if not annotations:
        print("No annotations provided.")
        return None

    res = annotations[0]

    if res.masks is None:
        print("No masks found.")
        return None

    # Original image and size
    img = res.orig_img
    orig_h, orig_w = img.shape[:2]
    original_area = orig_h * orig_w

    # Masks from YOLO (N, H', W')
    masks = res.masks.data.cpu().numpy()
    num_segments = masks.shape[0]

    # Compute area of each mask (in resized space)
    areas = [mask.sum() for mask in masks]

    # Index and mask with largest area
    largest_idx = int(np.argmax(areas))
    largest_mask = masks[largest_idx]

    # Rescale mask to original image size
    largest_mask_rescaled = cv2.resize(
        largest_mask.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )

    # Calculate mask area in original image space
    mask_area = largest_mask_rescaled.sum()
    mask_ratio = mask_area / original_area

    print(f"Detected {num_segments} masks. Largest mask index: {largest_idx}, area ratio: {mask_ratio:.2%}")

    # Fallback to original image if mask is less than 50% of original
    if mask_ratio < 0.35:
        print(f"Mask too small ({mask_ratio:.2%} < 35%). Using original image with normal resize.")
        upscaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        if visualize:
            plt.figure(figsize=(6, 9))
            plt.imshow(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
            plt.title("Original Image (Mask too small)")
            plt.axis("on")
            plt.show()
        return upscaled

    # Clean mask (assumes you have a clean_mask(mask) -> binary {0,1} function)
    largest_mask_clean = clean_mask(largest_mask_rescaled)
    
    # Expand the mask by 5% to add padding
    largest_mask_clean = expand_mask(largest_mask_clean, expand_percent=0.03)
    
    mask_clean = (largest_mask_clean * 255).astype(np.uint8)

    # Find contour + OBB on cleaned mask
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found in mask.")
        return None

    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)  # ((cx, cy), (w, h), angle)
    (cx, cy), (w, h), angle = rect
    print(w, h, angle)
    # Normalize so height is always the long side
    if w < h:
        upright_angle = angle
    else:
        upright_angle = angle - 90

    # Anti-rotate
    rotate_angle = upright_angle

    # Rotate the whole image around its center
    document_center = (cx, cy)
    M = cv2.getRotationMatrix2D(document_center, rotate_angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (orig_w, orig_h), flags=cv2.INTER_CUBIC)

    # Rotate mask similarly
    rotated_mask = cv2.warpAffine(mask_clean, M, (orig_w, orig_h), flags=cv2.INTER_NEAREST)

    # Find bounding rect on rotated mask (axis-aligned, since object is upright)
    contours2, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours2:
        print("No contours found in rotated mask.")
        return None

    cnt2 = contours2[0]
    x, y, bw, bh = cv2.boundingRect(cnt2)

    # Crop upright region
    cropped = rotated_img[y:y+bh, x:x+bw]

    # Apply zoom out to fit the cropped image onto target canvas at 90% fill
    upscaled = zoom_out_to_fit(cropped, target_w, target_h, fill_ratio=0.94)
    if visualize:
        # Show final result
        plt.figure(figsize=(6, 9))
        plt.imshow(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
        plt.title("Upright OBB Crop, Zoom-Out Fit (90%)")
        plt.axis("on")
        plt.show()
    return upscaled

def detect_and_crop(img_path):
  annotations = run_segment(img_path)
  return plot_segments(annotations)


def get_sorted_file_paths(dir_path):
    valid_exts = config.allowed_extensions

    files = [
        f for f in os.listdir(dir_path)
        if os.path.splitext(f.lower())[1] in valid_exts
    ]

    files_sorted = sorted(files)
    return [os.path.join(dir_path, f) for f in files_sorted]


def images_to_pdf(image_paths, output_pdf):
    # Open the first image
    if not image_paths:
        print("No images to convert.")
        return
    first = Image.open(image_paths[0]).convert("RGB")

    # Open the rest
    rest = [Image.open(p).convert("RGB") for p in image_paths[1:]]

    # Save as multi-page PDF
    first.save(output_pdf, save_all=True, append_images=rest)

def full_run(image_dir, output_pdf):
    image_paths = get_sorted_file_paths(image_dir)
    processed_image_paths = []
    original_image_paths = []
    target_width = config.target_width
    target_height = config.target_height

    for idx, img_path in enumerate(image_paths):
        print(f"Processing {img_path}...")
        
        # Save resized original for comparison PDF
        original_img = cv2.imread(img_path)
        if original_img is not None:
            original_resized = cv2.resize(original_img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            original_save_path = os.path.join("temp_processed", f"original_{idx+1}.jpg")
            os.makedirs("temp_processed", exist_ok=True)
            cv2.imwrite(original_save_path, original_resized)
            original_image_paths.append(original_save_path)
        
        try:
            cropped_img = detect_and_crop(img_path)
        except Exception as e: # fallback to original image on error
            cropped_img = cv2.imread(img_path)
            print(f"Error processing {img_path}: {e}. Using original image.")
            if cropped_img is None:
                print(f"Failed to read {img_path}. Skipping.")
                continue
            cropped_img = cv2.resize(cropped_img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    
        if cropped_img is not None:
            save_path = os.path.join("temp_processed", f"page_{idx+1}.jpg")
            os.makedirs("temp_processed", exist_ok=True)
            cv2.imwrite(save_path, cropped_img)
            processed_image_paths.append(save_path)
       
    if not processed_image_paths:
        print("No processed images to save.")
        raise ValueError("No processed images to save.")
    
    # Save processed PDF
    images_to_pdf(processed_image_paths, output_pdf)
    print(f"PDF saved to {output_pdf}")
    
    # Save original images PDF for comparison
    if original_image_paths:
        original_pdf_path = output_pdf.replace(".pdf", "_original.pdf")
        images_to_pdf(original_image_paths, original_pdf_path)
        print(f"Original PDF saved to {original_pdf_path}")


class PDFGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Generator")
        self.root.geometry("500x200")
        
        # Image Directory
        tk.Label(root, text="Image Directory:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.image_dir_var = tk.StringVar()
        tk.Entry(root, textvariable=self.image_dir_var, width=40).grid(row=0, column=1, padx=5, pady=10)
        tk.Button(root, text="Browse", command=self.browse_image_dir).grid(row=0, column=2, padx=5, pady=10)
        
        # Output Directory
        tk.Label(root, text="Output Directory:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.output_dir_var = tk.StringVar()
        tk.Entry(root, textvariable=self.output_dir_var, width=40).grid(row=1, column=1, padx=5, pady=10)
        tk.Button(root, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=10)
        
        # Output Filename
        tk.Label(root, text="Output Filename:").grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.output_name_var = tk.StringVar(value="output.pdf")
        tk.Entry(root, textvariable=self.output_name_var, width=40).grid(row=2, column=1, padx=5, pady=10)
        
        # Generate Button
        tk.Button(root, text="Generate PDF", command=self.generate_pdf, bg="green", fg="white").grid(row=3, column=1, pady=20)
    
    def browse_image_dir(self):
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            self.image_dir_var.set(directory)
    
    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
    
    def generate_pdf(self):
        image_dir = self.image_dir_var.get()
        output_dir = self.output_dir_var.get()
        output_name = self.output_name_var.get()
        
        if not image_dir:
            messagebox.showerror("Error", "Please select an image directory.")
            return
        if not output_dir:
            messagebox.showerror("Error", "Please select an output directory.")
            return
        if not output_name:
            messagebox.showerror("Error", "Please enter an output filename.")
            return
        
        if not output_name.endswith(".pdf"):
            output_name += ".pdf"
        
        output_pdf = os.path.join(output_dir, output_name)
        
        try:
            full_run(image_dir, output_pdf)
            messagebox.showinfo("Success", f"PDF saved to {output_pdf}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate PDF:\n{e}")
        
        
        

def main():
    root = tk.Tk()
    app = PDFGeneratorGUI(root)
    root.mainloop()
    
if __name__ == "__main__":
    main()