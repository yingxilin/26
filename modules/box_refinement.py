"""
Iterative Box Refinement Module
论文参考: RoBox-SAM (2024)
创新点: 利用 HQ-SAM 图像特征自动学习 YOLO bbox 的偏移量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
from typing import List, Tuple, Optional


class BoxEncoder(nn.Module):
    """将 bbox 坐标编码为高维特征向量"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 8维输入: [x1, y1, x2, y2, cx, cy, w, h] (归一化坐标)
        self.input_dim = 8
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU()
        )
        
        # 位置编码
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)
    
    def forward(self, bboxes, img_shape):
        """
        Args:
            bboxes: (B, 4) - [x1, y1, x2, y2] 像素坐标
            img_shape: (H, W)
        Returns:
            box_features: (B, hidden_dim)
        """
        B = bboxes.shape[0]
        H, W = img_shape
        
        # 归一化坐标到 [0, 1]
        normalized_boxes = bboxes.clone().float()
        normalized_boxes = normalized_boxes.clone()
        # 使用非原地操作以避免 autograd 冲突
        x_norm = normalized_boxes[:, [0, 2]] / W
        y_norm = normalized_boxes[:, [1, 3]] / H
        normalized_boxes = torch.stack(
            [x_norm[:, 0], y_norm[:, 0], x_norm[:, 1], y_norm[:, 1]], dim=1
        )
        
        # 计算中心点和宽高
        x1, y1, x2, y2 = normalized_boxes[:, 0], normalized_boxes[:, 1], normalized_boxes[:, 2], normalized_boxes[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        # 组合8维特征: [x1, y1, x2, y2, cx, cy, w, h]
        box_features = torch.stack([x1, y1, x2, y2, cx, cy, w, h], dim=1)  # (B, 8)
        
        # MLP编码
        encoded = self.mlp(box_features)  # (B, hidden_dim)
        
        # 添加位置编码
        encoded = self.pos_encoding(encoded)
        
        return encoded


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, x):
        # x: (B, d_model)
        B, d_model = x.shape
        position = torch.arange(d_model, device=x.device).float()
        
        # 计算位置编码
        div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(B, d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position[0::2] * div_term)
class OffsetPredictor(nn.Module):
    """预测 bbox 的偏移量"""
    
    def __init__(self, hidden_dim=256, max_offset=50):
        super().__init__()
        self.max_offset = max_offset
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # 增加dropout
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 4)  # 输出4个偏移值 [Δx1, Δy1, Δx2, Δy2]
        )
        
        # 初始化最后一层权重为小值
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.constant_(self.mlp[-1].bias, 0)near(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 输出4个偏移值 [Δx1, Δy1, Δx2, Δy2]
        )
    
    def forward(self, features):
        """
        Args:
            features: (B, hidden_dim)
        Returns:
            offsets: (B, 4) - [Δx1, Δy1, Δx2, Δy2] 像素单位
        """
        offsets = self.mlp(features)  # (B, 4)
   class BoxRefinementModule(nn.Module):
    """完整的 Box Refinement 模块"""
    
    def __init__(self, hidden_dim=256, num_heads=8, max_offset=50):
        super().__init__()
        self.box_encoder = BoxEncoder(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.offset_predictor = OffsetPredictor(hidden_dim, max_offset)
        self.hidden_dim = hidden_dim
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)_heads, batch_first=True)
        self.offset_predictor = OffsetPredictor(hidden_dim, max_offset)
        self.hidden_dim = hidden_dim
    
    def forward(self, image_embedding, bbox, img_shape):
        """
        单次前向传播 (用于一次迭代)
        
        Args:
            image_embedding: (B, 256, 64, 64) - HQ-SAM encoder 输出
            bbox: (B, 4) - 当前 bbox
            img_shape: (H, W)
        Returns:
            offset: (B, 4) - 预测的偏移量
        """
        B = bbox.shape[0]
        
        # 1. 编码 bbox
        box_features = self.box_encoder(bbox, img_shape)  # (B, hidden_dim)
        
        # 2. 展平 image_embedding 并进行 cross-attention
        # image_embedding: (B, 256, 64, 64) -> (B, 256, 4096)
        image_features = image_embedding.view(B, self.hidden_dim, -1)  # (B, 256, 4096)
        image_features = image_features.transpose(1, 2)  # (B, 4096, 256)
        
        # box_features: (B, hidden_dim) -> (B, 1, hidden_dim)
        box_features = box_features.unsqueeze(1)  # (B, 1, 256)
        
        # Cross-attention: Query=box, Key=Value=image
        refined_features, _ = self.cross_attention(
            query=box_features,      # (B, 1, 256)
            key=image_features,      # (B, 4096, 256)
            value=image_features     # (B, 4096, 256)
        )  # (B, 1, 256)
        
        # 3. 预测偏移量
        refined_features = refined_features.squeeze(1)  # (B, 256)
        offset = self.offset_predictor(refined_features)  # (B, 4)
        
        return offset
    
    def iterative_refine(self, image_embedding, initial_bbox, img_shape, 
                         max_iter=3, stop_threshold=1.0):
        """
        迭代精炼 bbox
        
        Args:
            image_embedding: (B, 256, 64, 64)
            initial_bbox: (B, 4) - YOLOv8 输出的 bbox
            img_shape: (H, W)
            max_iter: 最大迭代次数
            stop_threshold: 早停阈值 (像素)
        Returns:
            refined_bbox: (B, 4) - 精炼后的 bbox
            history: List[Tensor] - 每次迭代的 bbox (用于可视化)
        """
        B = initial_bbox.shape[0]
        H, W = img_shape
        
        # 初始化
        current_bbox = initial_bbox.clone()
        history = [current_bbox.clone()]
        
        for iter_idx in range(max_iter):
            # 预测偏移量
            offset = self.forward(image_embedding, current_bbox, img_shape)
            
            # 更新 bbox
            candidate_bbox = current_bbox + offset
            
            # 边界检查: 确保 bbox 在图像范围内（非原地操作）
            new_x1 = torch.clamp(candidate_bbox[:, 0], 0, W - 1)
            new_y1 = torch.clamp(candidate_bbox[:, 1], 0, H - 1)
            new_x2 = torch.clamp(candidate_bbox[:, 2], 0, W - 1)
            new_y2 = torch.clamp(candidate_bbox[:, 3], 0, H - 1)
            
            # 确保 x2 > x1, y2 > y1（非原地操作）
            new_x2 = torch.maximum(new_x2, new_x1 + 1)
            new_y2 = torch.maximum(new_y2, new_y1 + 1)
            
            new_bbox = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)
            
            # 记录历史
            history.append(new_bbox.clone())
            
            # 早停判断: 如果偏移量很小，提前停止
            offset_magnitude = torch.norm(offset, dim=1).mean()
            if offset_magnitude < stop_threshold:
                break
            
            # 更新当前 bbox
            current_bbox = new_bbox
        
        return current_bbox, history


def visualize_refinement(image, bbox_history, save_path, gt_bbox=None):
    """
    可视化 bbox 精炼过程
    
    Args:
        image: (H, W, 3) - 原始图像 (numpy array)
        bbox_history: List[Tensor] - 每次迭代的 bbox
        save_path: 保存路径
        gt_bbox: (4,) - Ground truth bbox (可选)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image)
    
    # 颜色映射: 红色(初始) -> 黄色(中间) -> 绿色(最终)
    colors = ['red', 'orange', 'yellow', 'green']
    
    for i, bbox in enumerate(bbox_history):
        if i >= len(colors):
            color = 'blue'
        else:
            color = colors[i]
        
        x1, y1, x2, y2 = bbox[0].cpu().numpy()
        width = x2 - x1
        height = y2 - y1
        
        # 绘制矩形
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none',
                               label=f'Iter {i}' if i < len(colors) else f'Iter {i}')
        ax.add_patch(rect)
        
        # 添加箭头显示移动方向 (除了最后一个)
        if i < len(bbox_history) - 1:
            next_bbox = bbox_history[i + 1][0].cpu().numpy()
            curr_center = [(x1 + x2) / 2, (y1 + y2) / 2]
            next_center = [(next_bbox[0] + next_bbox[2]) / 2, (next_bbox[1] + next_bbox[3]) / 2]
            
            ax.annotate('', xy=next_center, xytext=curr_center,
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    
    # 绘制 Ground Truth (如果提供)
    if gt_bbox is not None:
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
        gt_width = gt_x2 - gt_x1
        gt_height = gt_y2 - gt_y1
        gt_rect = patches.Rectangle((gt_x1, gt_y1), gt_width, gt_height,
                                  linewidth=3, edgecolor='purple', facecolor='none',
       def box_iou_loss(pred_boxes, target_boxes):
    """计算 bbox IoU loss (1 - IoU) - 修复版本"""
    # 确保输入张量在相同设备上
    if pred_boxes.device != target_boxes.device:
        target_boxes = target_boxes.to(pred_boxes.device)
    
    # 确保输入张量形状一致
    if pred_boxes.shape != target_boxes.shape:
        min_batch = min(pred_boxes.shape[0], target_boxes.shape[0])
        pred_boxes = pred_boxes[:min_batch]
        target_boxes = target_boxes[:min_batch]
    
    # 向量化计算IoU，避免循环
    # pred_boxes, target_boxes: (B, 4) - [x1, y1, x2, y2]
    
    # 计算交集坐标
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])  # (B,)
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])  # (B,)
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])  # (B,)
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])  # (B,)
    
    # 计算交集面积
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)  # (B,)
    
    # 计算各自面积
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])  # (B,)
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])  # (B,)
    
    # 计算并集面积
    union = pred_area + target_area - intersection  # (B,)
    
    # 计算IoU，添加数值稳定性
    iou = intersection / (union + 1e-7)  # (B,)
    
    # 返回平均IoU损失
    return 1.0 - iou.mean()  # 标量 
        iou = intersection / (union + 1e-6)
        return iou
    
    # 计算每个样本的IoU
    ious = []
    for i in range(pred_boxes.shape[0]):
        iou = compute_iou(pred_boxes[i], target_boxes[i])
        ious.append(iou)
    
    ious = torch.stack(ious)
    return 1.0 - ious.mean()  # 返回 1 - IoU 作为损失


if __name__ == "__main__":
    # 简单测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    B = 2
    H, W = 256, 256
    image_embedding = torch.randn(B, 256, 64, 64).to(device)
    initial_bbox = torch.tensor([[50, 50, 150, 150], [100, 100, 200, 200]], dtype=torch.float32).to(device)
    img_shape = (H, W)
    
    # 创建模型
    model = BoxRefinementModule().to(device)
    
    # 测试单次前向传播
    offset = model(image_embedding, initial_bbox, img_shape)
    print(f"Offset shape: {offset.shape}")
    print(f"Offset values: {offset}")
    
    # 测试迭代精炼
    refined_bbox, history = model.iterative_refine(image_embedding, initial_bbox, img_shape)
    print(f"Refined bbox shape: {refined_bbox.shape}")
    print(f"Number of iterations: {len(history)}")
    print("Test passed!")