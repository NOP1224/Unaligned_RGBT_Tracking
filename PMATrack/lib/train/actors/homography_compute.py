import torch
import numpy as np
import cv2

def make_M_flip_full_batch(W_full: torch.Tensor) -> torch.Tensor:
    # 返回 [B, 3, 3]
    B = W_full.shape[0]
    M = torch.zeros(B, 3, 3, device=W_full.device, dtype=torch.float32)
    M[:, 0, 0] = -1
    M[:, 1, 1] =  1
    M[:, 2, 2] =  1
    M[:, 0, 2] =  W_full
    return M

class HomographyBatch:
    def __init__(self, PATCH_W=256):
        self.PATCH_W = PATCH_W
        # 翻转矩阵（常量）
        self.M_flip_patch = np.array([
            [-1, 0, self.PATCH_W],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)


    # -------------------- 批量 DLT --------------------
    def batch_find_homography(self, pts_src, pts_dst):
        """
        批量 DLT 计算单应性矩阵
        pts_src: (B, 4, 2)
        pts_dst: (B, 4, 2)
        return: (B, 3, 3)
        """
        B, N, _ = pts_src.shape
        assert N >= 4, "至少需要4对点"

        Hs = []
        for b in range(B):
            src = pts_src[b]
            dst = pts_dst[b]
            A = []
            for i in range(N):
                x, y = src[i]
                u, v = dst[i]
                A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
                A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
            A = torch.tensor(A, dtype=torch.float32, device=pts_src.device)  # (8,9)
            _, _, V = torch.linalg.svd(A)
            h = V[-1, :]
            H = h.reshape(3, 3)
            H = H / H[2, 2]
            Hs.append(H)
        return torch.stack(Hs, dim=0)

    # -------------------- 主函数 --------------------
    def compute_batch(self, rgb_boxes, tir_boxes, jit_rgbs, crop_szs, 
                    is_flip_batch, is_flip_first_batch,
                    W_full: torch.Tensor, H_full: torch.Tensor):
        B = rgb_boxes.shape[0]
        W_full = W_full.to(rgb_boxes.device).float()
        H_full = H_full.to(rgb_boxes.device).float()
        
        # 1. scale tir -> rgb (批量)
        tir_boxes_scaled = torch.stack([
            tir_boxes[:, 0],
            tir_boxes[:, 1],
            tir_boxes[:, 2],
            tir_boxes[:, 3]
        ], dim=-1)

        # 2. crop box (批量)
        cxcy = jit_rgbs[:, 0:2] + 0.5 * jit_rgbs[:, 2:4]
        crop_boxes = torch.stack([
            cxcy[:, 0] - crop_szs / 2.0,
            cxcy[:, 1] - crop_szs / 2.0,
            crop_szs.float(),
            crop_szs.float()
        ], dim=-1)

        # 3. resize factor (批量)
        resize_factors = self.PATCH_W / crop_szs.float()

        # 4. 应用全图翻转 (批量)
        mask_first = is_flip_first_batch.bool()
        rgb_boxes_flipped = rgb_boxes.clone().float()
        tir_boxes_flipped = tir_boxes_scaled.clone().float()
        jit_rgbs_flipped = jit_rgbs.clone().float()

        rgb_boxes_flipped[mask_first, 0] = W_full[mask_first] - (rgb_boxes[mask_first, 0] + rgb_boxes[mask_first, 2])
        tir_boxes_flipped[mask_first, 0] = W_full[mask_first] - (tir_boxes[mask_first, 0] + tir_boxes[mask_first, 2])
        jit_rgbs_flipped[mask_first, 0]  = W_full[mask_first] - (jit_rgbs[mask_first, 0]  + jit_rgbs[mask_first, 2])

        # 5. 原图 -> patch (批量)
        def transform_batch(boxes: torch.Tensor, crop_boxes: torch.Tensor,
                            resize_factors: torch.Tensor, crop_szs: torch.Tensor,
                            normalize=False) -> torch.Tensor:
            """
            批量版 transform_image_to_crop
            args:
                boxes: (B,4)  [x,y,w,h]
                crop_boxes: (B,4) [x,y,w,h]
                resize_factors: (B,)
                crop_szs: (B,)   (裁剪后的 patch 尺寸，假设方形)
                normalize: bool
            returns:
                (B,4) [x,y,w,h]  在 patch 上的 box
            """
            # 1. 中心点
            crop_centers = crop_boxes[:, 0:2] + 0.5 * crop_boxes[:, 2:4]   # (B,2)
            box_centers  = boxes[:, 0:2] + 0.5 * boxes[:, 2:4]             # (B,2)

            # 2. 变换到 patch 坐标
            crop_half = (crop_szs - 1).unsqueeze(1) / 2  # (B,1) -> (B,2)
            box_out_centers = crop_half + (box_centers - crop_centers) * resize_factors.unsqueeze(1)  # (B,2)

            # 3. 尺寸缩放
            box_out_wh = boxes[:, 2:4] * resize_factors.unsqueeze(1)  # (B,2)

            # 4. 拼回 [x,y,w,h]
            box_out_xy = box_out_centers - 0.5 * box_out_wh
            box_out = torch.cat([box_out_xy, box_out_wh], dim=-1)

            if normalize:
                box_out = box_out / crop_szs.unsqueeze(1)  # 归一化到 [0,1]
            return box_out

        crop_sz_tensor = torch.full((B,), self.PATCH_W, dtype=torch.float32, device=rgb_boxes.device)  # (B,)
        rgb_in_patch = transform_batch(rgb_boxes_flipped, crop_boxes, resize_factors, crop_sz_tensor)
        tir_in_patch = transform_batch(tir_boxes_flipped, crop_boxes, resize_factors, crop_sz_tensor)

        # 6. patch 翻转 (批量)
        mask_patch = is_flip_batch.bool()
        rgb_in_patch[mask_patch, 0] = self.PATCH_W - (rgb_in_patch[mask_patch, 0] + rgb_in_patch[mask_patch, 2])
        tir_in_patch[mask_patch, 0] = self.PATCH_W - (tir_in_patch[mask_patch, 0] + tir_in_patch[mask_patch, 2])

        # 7. 构造四点 (批量 numpy)
        def box_to_pts(boxes: torch.Tensor):
            """
            boxes: (B,4) torch.Tensor [x,y,w,h] (可能在 GPU 上)
            return: (B,4,2) numpy float32
            """
            boxes = boxes.detach().cpu().numpy()  # ✅ 先转到 CPU
            x1, y1, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            return np.stack([
                np.stack([x1, y1], axis=-1),
                np.stack([x1 + w, y1], axis=-1),
                np.stack([x1 + w, y1 + h], axis=-1),
                np.stack([x1, y1 + h], axis=-1)
            ], axis=1).astype(np.float32)


        pts_rgb = box_to_pts(rgb_in_patch)
        pts_tir = box_to_pts(tir_in_patch)

        # 8. 逐个求解 Homography (循环不可避免)
        H_full_batch = []
        H_orig_batch = []
        for idx in range(B):
            H_orig, status = cv2.findHomography(pts_tir[idx], pts_rgb[idx],
                                                method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if H_orig is None:
                H_full_batch.append(None)
                continue
            else:
                H_orig = H_orig.astype(np.float32)   
                H_orig_batch.append(H_orig)

            # crop -> patch 矩阵
            x_crop, y_crop, w_crop, h_crop = crop_boxes[idx].tolist()
            scale = self.PATCH_W / w_crop
            T_crop = np.array([
                [scale, 0, -scale * x_crop],
                [0, scale, -scale * y_crop],
                [0, 0, 1]
            ], dtype=np.float32)
            T_crop_inv = np.linalg.inv(T_crop)
            H_proc = H_orig.copy()
            if mask_patch[idx].item():
                H_proc = self.M_flip_patch @ H_proc @ self.M_flip_patch


            H_full = T_crop_inv @ H_proc @ T_crop

            # if mask_first[idx].item():
            #     H_full = self.M_flip_full @ H_full @ self.M_flip_full
            if mask_first[idx].item():
                W_i = float(W_full[idx].item())
                M_flip_full_i = np.array([[-1, 0, W_i],
                                        [ 0, 1, 0 ],
                                        [ 0, 0, 1 ]], dtype=np.float32)
                H_full = M_flip_full_i @ H_full @ M_flip_full_i 
                
            H_full = H_full.astype(np.float32)  
            H_full_batch.append(H_full)
            
        return H_full_batch, H_orig_batch, rgb_in_patch, tir_in_patch
    
    def local_to_global(self, H_local, crop_box, is_flip_patch=False, is_flip_full=False, W_full=None):
        x_crop, y_crop, w_crop, h_crop = crop_box
        scale = self.PATCH_W / w_crop
        T_crop = np.array([[scale, 0, -scale * x_crop],
                        [0, scale, -scale * y_crop],
                        [0, 0, 1]], dtype=np.float32)
        T_crop_inv = np.linalg.inv(T_crop)

        H_proc = H_local.copy()
        if is_flip_patch:
            H_proc = self.M_flip_patch @ H_proc @ self.M_flip_patch

        H_global = T_crop_inv @ H_proc @ T_crop

        if is_flip_full:
            assert W_full is not None, "需要该图像的宽度 W_full"
            M_flip_full = np.array([[-1, 0, W_full],
                                    [ 0, 1, 0     ],
                                    [ 0, 0, 1     ]], dtype=np.float32)
            H_global = M_flip_full @ H_global @ M_flip_full

        return H_global.astype(np.float32)

    
import math

class HomographyNormalizer:
    def __init__(self, patch_w, patch_h, margin_ratio=0.1):
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.margin_x = patch_w * margin_ratio
        self.margin_y = patch_h * margin_ratio

        Wn = patch_w + 2 * self.margin_x
        Hn = patch_h + 2 * self.margin_y

        # 归一化矩阵
        self.T = torch.tensor([
            [2.0/Wn, 0, -(1 + 2*self.margin_x/Wn)],
            [0, 2.0/Hn, -(1 + 2*self.margin_y/Hn)],
            [0, 0, 1]
        ], dtype=torch.float32)

        self.T_inv = torch.inverse(self.T)

    def normalize(self, H):
        """ H: [B, 3, 3], 原始像素尺度 """
        B = H.shape[0]
        T = self.T.to(H.device).unsqueeze(0).expand(B, -1, -1)
        T_inv = self.T_inv.to(H.device).unsqueeze(0).expand(B, -1, -1)
        return torch.bmm(torch.bmm(T, H), T_inv)

    def denormalize(self, Hn, clip_range=None):
        """ Hn: [B, 3, 3], 归一化后的
            clip_range: tuple((x_min, x_max), (y_min, y_max)), 可选
        """
        B = Hn.shape[0]
        T = self.T.to(Hn.device).unsqueeze(0).expand(B, -1, -1)
        T_inv = self.T_inv.to(Hn.device).unsqueeze(0).expand(B, -1, -1)
        H = torch.bmm(torch.bmm(T_inv, Hn), T)
        
        if clip_range is not None:
            (x_min, x_max), (y_min, y_max) = clip_range
            # 使用 out-of-place 的方式
            H = H.clone()  # 可选，确保不修改原始张量
            H0_2 = H[:, 0, 2].clamp(x_min, x_max)
            H1_2 = H[:, 1, 2].clamp(y_min, y_max)
            H = H.clone()  # 确保新的张量用于梯度计算
            H[:, 0, 2] = H0_2
            H[:, 1, 2] = H1_2
            H = H.clone()
            H[:,1,0] = 0
        return H
    
    

class SimpleHomoNormalizer:
    def __init__(self, patch_w, patch_h, margin_ratio=0.1):
        self.patch_w = patch_w
        self.patch_h = patch_h

    def normalize(self, H: torch.Tensor):
        """ H: [B, 3, 3] 或 [3, 3]，Torch Tensor """
        sx, sy = H[:, 0, 0], H[:, 1, 1]
        deltax, deltay = H[:, 0, 2], H[:, 1, 2]

        # 归一化
        deltax_norm = deltax / self.patch_w
        deltay_norm = deltay / self.patch_h

        # 保持 Torch Tensor，不用 numpy
        H_norm = torch.zeros_like(H)
        H_norm[:, 0, 0] = torch.log(sx)
        H_norm[:, 1, 1] = torch.log(sy)
        H_norm[:, 0, 2] = deltax_norm
        H_norm[:, 1, 2] = deltay_norm
        H_norm[:, 2, 2] = 1.0
        return H_norm
    
    def denormalize(self, H_norm: torch.Tensor, inverse: bool=False):
        # 取 log-scale 与 norm 平移
        sx_log, sy_log = H_norm[:, 0, 0], H_norm[:, 1, 1]
        deltax_norm, deltay_norm = H_norm[:, 0, 2], H_norm[:, 1, 2]

        # exp 还原尺度
        sx = torch.exp(sx_log)
        sy = torch.exp(sy_log)

        deltax = deltax_norm * self.patch_w
        deltay = deltay_norm * self.patch_h

        if inverse:
            sx = 1.0 / sx
            sy = 1.0 / sy
            deltax = -deltax / sx
            deltay = -deltay / sy

        # 组装矩阵
        H = torch.zeros_like(H_norm)
        H[:, 0, 0] = sx
        H[:, 1, 1] = sy
        H[:, 0, 2] = deltax
        H[:, 1, 2] = deltay
        H[:, 2, 2] = 1.0
        return H
