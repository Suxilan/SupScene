import numpy as np
import cv2
from PIL import Image
import matplotlib.cm as cm # 引入 Matplotlib 的色彩映射

def create_overlay_batch(original_images: np.ndarray, 
                           attention_maps: np.ndarray, 
                           alpha: float = 0.6, 
                           colormap_name: str = 'viridis') -> np.ndarray:
    """
    【批量处理版】将注意力图批量叠加到原始图像上，并返回一个图像批次。

    Args:
        original_images (np.ndarray): 原始彩色图像批次，格式为 (B, 3, H, W)，
                                      范围 [0, 255], uint8。
        attention_maps (np.ndarray): 注意力图批次，格式为 (B, K, H_map, W_map)，
                                     范围 [0, 1], float。
        alpha (float, optional): 热力图的不透明度。默认为 0.6。
        colormap_name (str, optional): Matplotlib colormap 名称。默认为 'viridis'。

    Returns:
        np.ndarray: 叠加后的图像批次，格式为 (B, 3, H, W)，范围 [0, 255], uint8。
    """
    batch_size, _, img_h, img_w = original_images.shape
    overlay_batch = []
    
    cmap = cm.get_cmap(colormap_name)

    for i in range(batch_size):
        # 1. 准备单张原始图像 (从 NumPy (C,H,W) -> Pillow RGBA)
        single_image_chw = original_images[i]
        image_rgb = single_image_chw.transpose(1, 2, 0)
        original_pil = Image.fromarray(image_rgb).convert('RGBA')

        # 2. 准备单张热力图
        # a. 合并 K 个注意力图
        single_attention_maps_k_hw = attention_maps[i]
        combined_map = np.max(single_attention_maps_k_hw, axis=0)
        
        # b. 夸大注意力图以便显示（由于每个H,W维度和为1，需要增强对比度）
        # 使用平方根或幂次方法增强对比度
        # enhanced_map = np.power(combined_map, 0.5)  # 平方根增强
        # 或者使用线性拉伸到[0,1]范围
        if combined_map.max() > combined_map.min():
            enhanced_map = (combined_map - combined_map.min()) / (combined_map.max() - combined_map.min())
        
        # c. 缩放并应用色彩映射
        resized_map = cv2.resize(enhanced_map, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        heatmap_rgba = cmap(resized_map, bytes=False)
        
        # d. 控制透明度
        heatmap_rgba[..., 3] = alpha * heatmap_rgba[..., 3]
        
        # e. 转换为 Pillow 图像
        heatmap_pil = Image.fromarray((heatmap_rgba * 255).astype(np.uint8), 'RGBA')

        # 3. 高质量叠加
        overlayed_pil = Image.alpha_composite(original_pil, heatmap_pil)

        # 4. 转换回 NumPy (H,W,C) -> (C,H,W)
        overlayed_rgb_hwc = np.array(overlayed_pil.convert('RGB'))
        overlayed_rgb_chw = overlayed_rgb_hwc.transpose(2, 0, 1)
        
        overlay_batch.append(overlayed_rgb_chw)

    # 5. 将处理完的图像列表堆叠成一个新的批次
    return np.stack(overlay_batch, axis=0)

if __name__ == '__main__':
    # --- 用于测试批量处理函数的示例 ---
    
    B_test = 4  # 批次大小
    K_test = 4  # 高斯核数量
    H_img_test, W_img_test = 256, 256
    H_map_test, W_map_test = 32, 32

    # 1. 模拟生成一个批次的原始图像
    # (B, H, W, C) -> (B, C, H, W)
    # 读取四次，拼成一个batch
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
    
    # 2. 生成真实的高斯核注意力图
    attention_maps_batch = []
    for b in range(B_test):
        batch_maps = []
        for k in range(K_test):
            # 随机生成高斯核参数
            mu_x = np.random.uniform(0.2, 0.8) * W_map_test  # 中心x坐标
            mu_y = np.random.uniform(0.2, 0.8) * H_map_test  # 中心y坐标
            sigma_x = np.random.uniform(3, 8)  # x方向标准差
            sigma_y = np.random.uniform(3, 8)  # y方向标准差
            theta = np.random.uniform(-np.pi, np.pi)  # 旋转角度
            
            # 生成坐标网格
            y_coords, x_coords = np.meshgrid(np.arange(H_map_test), np.arange(W_map_test), indexing='ij')
            
            # 计算旋转后的坐标
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            dx = x_coords - mu_x
            dy = y_coords - mu_y
            dx_rot = cos_theta * dx + sin_theta * dy
            dy_rot = -sin_theta * dx + cos_theta * dy
            
            # 计算高斯权重
            gaussian_map = np.exp(-0.5 * ((dx_rot / sigma_x)**2 + (dy_rot / sigma_y)**2))
            batch_maps.append(gaussian_map)
        
        attention_maps_batch.append(np.stack(batch_maps, axis=0))
    
    attention_maps_batch = np.stack(attention_maps_batch, axis=0).astype(np.float32)
    # 验证注意力图的归一化
    attention_sums = attention_maps_batch.sum(axis=(2, 3))  # [B, K] 每个高斯核的空间权重和
    print(f"注意力图权重和: {attention_sums}")
    print(f"权重和范围: [{attention_sums.min():.4f}, {attention_sums.max():.4f}]")
    
    # 归一化注意力图使其和为1
    attention_maps_batch = attention_maps_batch / (attention_sums[:, :, np.newaxis, np.newaxis] + 1e-6)
    print(attention_maps_batch[0])
    # 验证归一化后的和
    normalized_sums = attention_maps_batch.sum(axis=(2, 3))
    print(f"归一化后权重和: {normalized_sums}")
    is_normalized = np.allclose(normalized_sums, 1.0, atol=1e-5)
    print(f"权重和是否≈1: {is_normalized}")

    print(f"输入图像维度: {original_images_batch_chw.shape}")
    print(f"输入注意力图维度: {attention_maps_batch.shape}")

    # --- 调用批量处理函数 ---
    overlay_result_batch = create_overlay_batch(
        original_images=original_images_batch_chw,
        attention_maps=attention_maps_batch,
        alpha=0.7,
        colormap_name='magma'#'plasma'，'viridis'，'inferno'，'magma'
    )

    print(f"输出叠加图维度: {overlay_result_batch.shape}")

    # 验证维度是否正确
    assert overlay_result_batch.shape == original_images_batch_chw.shape

    # 保存批次中的第一张图以供查看
    first_overlay_image_chw = overlay_result_batch[0]
    first_overlay_image_hwc = first_overlay_image_chw.transpose(1, 2, 0)
    # Pillow 保存的是 RGB, OpenCV 保存需要转为 BGR
    first_overlay_image_bgr = cv2.cvtColor(first_overlay_image_hwc, cv2.COLOR_RGB2BGR)
    
    save_path_test = "attention_overlay_batch_test.png"
    cv2.imwrite(save_path_test, first_overlay_image_bgr)
    print(f"✅ 批量处理成功, 第一张样本图已保存至: {save_path_test}")