import cv2
import numpy as np
def align_targets(six_channel_img, rgb_bbox, ir_bbox_original):
    """
    参数说明：
    six_channel_img: 输入的六通道图像 (1080, 1920, 6)
    rgb_bbox: RGB图像的目标框 (x, y, w, h) 基于1920x1080
    ir_bbox_original: 红外图像原始尺寸的目标框 (x, y, w, h) 基于640x512
    
    返回：
    对齐后的六通道图像
    """
    # 分离RGB和红外通道
    rgb = six_channel_img[:, :, :3].copy()
    ir = six_channel_img[:, :, 3:6].copy()

    # 转换红外框到resize后的坐标系
    scale_x = 1920 / 640  # 宽度缩放比例
    scale_y = 1080 / 512  # 高度缩放比例
    
    # 将原始红外框转换到放大后的坐标系
    x_ir = ir_bbox_original[0] * scale_x
    y_ir = ir_bbox_original[1] * scale_y
    w_ir = ir_bbox_original[2] * scale_x
    h_ir = ir_bbox_original[3] * scale_y

    # 计算中心点坐标
    rgb_cx = rgb_bbox[0] + rgb_bbox[2]/2
    rgb_cy = rgb_bbox[1] + rgb_bbox[3]/2
    ir_cx = x_ir + w_ir/2
    ir_cy = y_ir + h_ir/2

    # 计算需要的缩放比例（保持原始红外目标尺寸）
    scale_x = rgb_bbox[2] / w_ir  # 宽度缩放比例
    scale_y = rgb_bbox[3] / h_ir  # 高度缩放比例

    # 构造仿射变换矩阵
    M = np.array([
        [scale_x, 0, rgb_cx - ir_cx * scale_x],
        [0, scale_y, rgb_cy - ir_cy * scale_y]
    ], dtype=np.float32)

    # 对红外图像进行仿射变换
    aligned_ir = cv2.warpAffine(
        ir, M, (1920, 1080),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # 合并通道并返回结果
    return np.concatenate([rgb, aligned_ir], axis=2)