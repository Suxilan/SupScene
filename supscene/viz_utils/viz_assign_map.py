import numpy as np
import cv2
from PIL import Image
import matplotlib.cm as cm 

def create_overlay_batch(original_images: np.ndarray, 
                         attention_maps: np.ndarray, 
                         alpha: float = 0.6, 
                         colormap_name: str = 'viridis') -> np.ndarray:
    """
    Batch version: overlay attention maps onto original images and return a batch of images.

    Args:
        original_images (np.ndarray): Batch of original color images, shape (B, 3, H, W),
                                      range [0, 255], dtype uint8.
        attention_maps (np.ndarray): Batch of attention maps, shape (B, K, H_map, W_map),
                                     range [0, 1], dtype float.
        alpha (float, optional): Opacity for the heatmap. Defaults to 0.6.
        colormap_name (str, optional): Matplotlib colormap name. Defaults to 'viridis'.

    Returns:
        np.ndarray: Batch of overlayed images, shape (B, 3, H, W), range [0, 255], dtype uint8.
    """
    batch_size, _, img_h, img_w = original_images.shape
    overlay_batch = []
    
    cmap = cm.get_cmap(colormap_name)

    for i in range(batch_size):
        # 1. Prepare a single original image (NumPy (C,H,W) -> Pillow RGBA)
        single_image_chw = original_images[i]
        image_rgb = single_image_chw.transpose(1, 2, 0)
        original_pil = Image.fromarray(image_rgb).convert('RGBA')

        # 2. Prepare a single heatmap
        # a. Combine K attention maps by taking the max
        single_attention_maps_k_hw = attention_maps[i]
        combined_map = np.max(single_attention_maps_k_hw, axis=0)
        
        # b. Amplify the attention map for better visibility (each H,W sum may be small)
        # Use square root or power methods to enhance contrast
        # enhanced_map = np.power(combined_map, 0.5)  # square-root enhancement
        # Or use linear stretching to [0,1]
        if combined_map.max() > combined_map.min():
            enhanced_map = (combined_map - combined_map.min()) / (combined_map.max() - combined_map.min())
        
        # c. Resize and apply colormap
        resized_map = cv2.resize(enhanced_map, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        heatmap_rgba = cmap(resized_map, bytes=False)
        
        # d. Control alpha channel
        heatmap_rgba[..., 3] = alpha * heatmap_rgba[..., 3]
        
        # e. Convert to Pillow image
        heatmap_pil = Image.fromarray((heatmap_rgba * 255).astype(np.uint8), 'RGBA')

        # 3. High-quality alpha composite
        overlayed_pil = Image.alpha_composite(original_pil, heatmap_pil)

        # 4. Convert back to NumPy (H,W,C) -> (C,H,W)
        overlayed_rgb_hwc = np.array(overlayed_pil.convert('RGB'))
        overlayed_rgb_chw = overlayed_rgb_hwc.transpose(2, 0, 1)
        
        overlay_batch.append(overlayed_rgb_chw)

    # 5. Stack the processed images into a new batch
    return np.stack(overlay_batch, axis=0)

if __name__ == '__main__':
    
    B_test = 4  # batch size
    K_test = 4  # number of Gaussian kernels
    H_img_test, W_img_test = 256, 256
    H_map_test, W_map_test = 32, 32

    # 1. Simulate reading a batch of original images
    # (B, H, W, C) -> (B, C, H, W)
    image_paths = [
        "data/GL3D/000000000000000000000000/images/00000003.jpg",
        "data/GL3D/000000000000000000000000/images/00000004.jpg", 
        "data/GL3D/000000000000000000000000/images/00000005.jpg",
        "data/GL3D/000000000000000000000000/images/00000006.jpg"
    ]
    
    original_images_batch = []
    for path in image_paths:
        color_img = cv2.imread(path, cv2.IMREAD_COLOR)
        if color_img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        # BGR -> RGB -> (H,W,C) -> (C,H,W)
        rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        chw_img = rgb_img.transpose(2, 0, 1)
        original_images_batch.append(chw_img)
    
    original_images_batch_chw = np.stack(original_images_batch, axis=0)
    H_img_test, W_img_test = original_images_batch_chw.shape[2:]
    
    # 2. Generate realistic Gaussian attention maps
    attention_maps_batch = []
    for b in range(B_test):
        batch_maps = []
        for k in range(K_test):
            # Random Gaussian parameters
            mu_x = np.random.uniform(0.2, 0.8) * W_map_test  # center x
            mu_y = np.random.uniform(0.2, 0.8) * H_map_test  # center y
            sigma_x = np.random.uniform(3, 8)  # stddev in x
            sigma_y = np.random.uniform(3, 8)  # stddev in y
            theta = np.random.uniform(-np.pi, np.pi)  # rotation angle
            
            # Create coordinate grid
            y_coords, x_coords = np.meshgrid(np.arange(H_map_test), np.arange(W_map_test), indexing='ij')
            
            # Compute rotated coordinates
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            dx = x_coords - mu_x
            dy = y_coords - mu_y
            dx_rot = cos_theta * dx + sin_theta * dy
            dy_rot = -sin_theta * dx + cos_theta * dy
            
            # Compute Gaussian weights
            gaussian_map = np.exp(-0.5 * ((dx_rot / sigma_x)**2 + (dy_rot / sigma_y)**2))
            batch_maps.append(gaussian_map)
        
        attention_maps_batch.append(np.stack(batch_maps, axis=0))
    
    attention_maps_batch = np.stack(attention_maps_batch, axis=0).astype(np.float32)
    # Verify attention map sums
    attention_sums = attention_maps_batch.sum(axis=(2, 3))  # [B, K] spatial sums per Gaussian
    print(f"Attention map sums: {attention_sums}")
    print(f"Sum range: [{attention_sums.min():.4f}, {attention_sums.max():.4f}]")
    
    # Normalize attention maps so each kernel sums to 1
    attention_maps_batch = attention_maps_batch / (attention_sums[:, :, np.newaxis, np.newaxis] + 1e-6)
    print(attention_maps_batch[0])
    # Verify normalization
    normalized_sums = attention_maps_batch.sum(axis=(2, 3))
    print(f"Normalized sums: {normalized_sums}")
    is_normalized = np.allclose(normalized_sums, 1.0, atol=1e-5)
    print(f"Sums approximately 1: {is_normalized}")

    print(f"Input image shape: {original_images_batch_chw.shape}")
    print(f"Input attention map shape: {attention_maps_batch.shape}")

    # --- Call the batch overlay function ---
    overlay_result_batch = create_overlay_batch(
        original_images=original_images_batch_chw,
        attention_maps=attention_maps_batch,
        alpha=0.7,
        colormap_name='magma'  # options: 'plasma', 'viridis', 'inferno', 'magma'
    )

    print(f"Output overlay batch shape: {overlay_result_batch.shape}")

    # Verify shapes match
    assert overlay_result_batch.shape == original_images_batch_chw.shape

    # Save the first overlaid image for inspection
    first_overlay_image_chw = overlay_result_batch[0]
    first_overlay_image_hwc = first_overlay_image_chw.transpose(1, 2, 0)
    # Pillow uses RGB; OpenCV expects BGR for saving
    first_overlay_image_bgr = cv2.cvtColor(first_overlay_image_hwc, cv2.COLOR_RGB2BGR)
    
    save_path_test = "attention_overlay_batch_test.png"
    cv2.imwrite(save_path_test, first_overlay_image_bgr)
    print(f"✅ Batch processing succeeded, first sample saved to: {save_path_test}")
