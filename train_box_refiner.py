"""
训练 Box Refinement Module
监督信号: Ground Truth Mask 的最小外接矩形 (作为 target bbox)
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加modules目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.box_refinement import BoxRefinementModule, box_iou_loss, visualize_refinement
from modules.hqsam_feature_extractor import create_hqsam_extractor


class FungiDataset(Dataset):
    """FungiTastic 数据集加载器"""
    
    def __init__(self, data_root: str, split: str = 'train', 
                 image_size: int = 300, data_subset: str = 'Mini',
                 augmentation: bool = True, debug: bool = False):
        """
        Args:
            data_root: 数据集根目录
            split: 'train' 或 'val'
            image_size: 图像尺寸
            data_subset: 'Mini' 或 'Full'
            augmentation: 是否启用数据增强
            debug: 调试模式
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.data_subset = data_subset
        self.augmentation = augmentation
        self.debug = debug
        
        # 构建路径
        self.images_dir = self.data_root / f"{data_subset}" / split / f"{image_size}p"
        self.masks_dir = self.data_root / f"{data_subset}" / split / "masks"
        
        # 检查路径是否存在
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        
        # 获取图像文件列表
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) + 
                                 list(self.images_dir.glob("*.JPG")) + 
                                 list(self.images_dir.glob("*.png")))
        
        if debug:
            self.image_files = self.image_files[:100]  # 调试模式只使用前100张图像
        
        print(f"Found {len(self.image_files)} images in {self.split} split")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载对应的mask
        mask_path = self.masks_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            # 如果没有对应的mask，创建一个空的
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 调整图像尺寸
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # 从mask计算ground truth bbox
        gt_bbox = self._compute_bbox_from_mask(mask)
        
        # 生成noisy bbox (模拟YOLO输出)
        noisy_bbox = self._generate_noisy_bbox(gt_bbox, image.shape[:2])
        
        return {
            'image': image,
            'mask': mask,
            'gt_bbox': gt_bbox,
            'noisy_bbox': noisy_bbox,
            'image_path': str(image_path)
        }
    
    def _compute_bbox_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """从mask计算最小外接矩形"""
        # 找到非零像素
        coords = np.column_stack(np.where(mask > 0))
        
        if len(coords) == 0:
            # 如果没有前景像素，返回中心的小矩形
            h, w = mask.shape
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 10
            return np.array([center_x - size, center_y - size, 
                           center_x + size, center_y + size], dtype=np.float32)
        
        # 计算边界框
        x1, y1 = coords.min(axis=0)[::-1]  # 注意坐标顺序
        x2, y2 = coords.max(axis=0)[::-1]
        
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    
    def _generate_noisy_bbox(self, gt_bbox: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """生成带噪声的bbox (模拟YOLO输出)"""
        h, w = img_shape
        
        # 添加随机偏移和缩放
        noise_scale = 0.1  # 10%的噪声
        offset_scale = 20   # 最大20像素的偏移
        
        x1, y1, x2, y2 = gt_bbox
        
        # 计算中心点和尺寸
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = x2 - x1
        bh = y2 - y1
        
        # 添加噪声
        cx_noise = np.random.normal(0, offset_scale)
        cy_noise = np.random.normal(0, offset_scale)
        scale_noise = np.random.normal(1.0, noise_scale)
        
        # 应用噪声
        new_cx = cx + cx_noise
        new_cy = cy + cy_noise
        new_bw = bw * scale_noise
        new_bh = bh * scale_noise
        
        # 计算新的bbox
        new_x1 = max(0, new_cx - new_bw / 2)
        new_y1 = max(0, new_cy - new_bh / 2)
        new_x2 = min(w - 1, new_cx + new_bw / 2)
        new_y2 = min(h - 1, new_cy + new_bh / 2)
        
        return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)


def collate_fn(batch):
    """自定义collate函数"""
    images = torch.stack([torch.from_numpy(item['image']).permute(2, 0, 1).float() / 255.0 
                         for item in batch])
    masks = torch.stack([torch.from_numpy(item['mask']).float() for item in batch])
    gt_bboxes = torch.stack([torch.from_numpy(item['gt_bbox']).float() for item in batch])
    noisy_bboxes = torch.stack([torch.from_numpy(item['noisy_bbox']).float() for item in batch])
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'masks': masks,
        'gt_bboxes': gt_bboxes,
        'noisy_bboxes': noisy_bboxes,
        'image_paths': image_paths
    }


def compute_loss(pred_bboxes, gt_bboxes, l1_weight=1.0, iou_weight=2.0):
    """计算损失函数"""
    # L1损失
    l1_loss = nn.L1Loss()(pred_bboxes, gt_bboxes)
    
    # IoU损失
    iou_loss = box_iou_loss(pred_bboxes, gt_bboxes)
    
    # 总损失
    total_loss = l1_weight * l1_loss + iou_weight * iou_loss
    
    return total_loss, l1_loss, iou_loss


def train_one_epoch(model, dataloader, optimizer, hqsam_extractor, device, epoch, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_l1_loss = 0
    total_iou_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        gt_bboxes = batch['gt_bboxes'].to(device)
        noisy_bboxes = batch['noisy_bboxes'].to(device)
        image_paths = batch['image_paths']
        
        # 提取图像特征
        image_features_list = []
        for i in range(images.shape[0]):
            image_np = images[i].permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            features = hqsam_extractor.extract_features(image_np)
            image_features_list.append(features)
        
        # 堆叠特征
        image_features = torch.cat(image_features_list, dim=0)  # (B, 256, 64, 64)
        
        # 前向传播
        optimizer.zero_grad()
        
        # 迭代精炼
        refined_bboxes, history = model.iterative_refine(
            image_features, noisy_bboxes, (config['data']['image_size'], config['data']['image_size']),
            max_iter=config['refinement']['max_iter'],
            stop_threshold=config['refinement']['stop_threshold']
        )
        
        # 计算损失
        loss, l1_loss, iou_loss = compute_loss(
            refined_bboxes, gt_bboxes,
            l1_weight=config['loss']['l1_weight'],
            iou_weight=config['loss']['iou_weight']
        )
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 更新统计
        total_loss += loss.item()
        total_l1_loss += l1_loss.item()
        total_iou_loss += iou_loss.item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'L1': f'{l1_loss.item():.4f}',
            'IoU': f'{iou_loss.item():.4f}'
        })
        
        # 可视化（可选）
        if batch_idx % config['output']['vis_freq'] == 0:
            visualize_batch(model, batch, image_features, hqsam_extractor, 
                          config, epoch, batch_idx)
    
    return total_loss / len(dataloader), total_l1_loss / len(dataloader), total_iou_loss / len(dataloader)


def visualize_batch(model, batch, image_features, hqsam_extractor, config, epoch, batch_idx):
    """可视化一个batch的结果"""
    device = next(model.parameters()).device
    
    # 选择第一个样本进行可视化
    image = batch['images'][0].permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    gt_bbox = batch['gt_bboxes'][0].cpu().numpy()
    noisy_bbox = batch['noisy_bboxes'][0].cpu().numpy()
    
    # 获取精炼后的bbox
    with torch.no_grad():
        refined_bbox, history = model.iterative_refine(
            image_features[0:1], noisy_bbox[None], 
            (config['data']['image_size'], config['data']['image_size']),
            max_iter=config['refinement']['max_iter']
        )
        refined_bbox = refined_bbox[0].cpu().numpy()
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像 + GT bbox
    axes[0].imshow(image)
    gt_rect = plt.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1],
                           linewidth=2, edgecolor='green', facecolor='none', label='GT')
    axes[0].add_patch(gt_rect)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    # 原始图像 + Noisy bbox
    axes[1].imshow(image)
    noisy_rect = plt.Rectangle((noisy_bbox[0], noisy_bbox[1]), noisy_bbox[2]-noisy_bbox[0], noisy_bbox[3]-noisy_bbox[1],
                              linewidth=2, edgecolor='red', facecolor='none', label='Noisy')
    axes[1].add_patch(noisy_rect)
    axes[1].set_title('Noisy (YOLO)')
    axes[1].axis('off')
    
    # 原始图像 + Refined bbox
    axes[2].imshow(image)
    refined_rect = plt.Rectangle((refined_bbox[0], refined_bbox[1]), refined_bbox[2]-refined_bbox[0], refined_bbox[3]-refined_bbox[1],
                                linewidth=2, edgecolor='blue', facecolor='none', label='Refined')
    axes[2].add_patch(refined_rect)
    axes[2].set_title('Refined')
    axes[2].axis('off')
    
    plt.suptitle(f'Epoch {epoch}, Batch {batch_idx}')
    plt.tight_layout()
    
    # 保存图像
    vis_dir = Path(config['output']['visualization_dir'])
    vis_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(vis_dir / f'epoch_{epoch}_batch_{batch_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()


def evaluate(model, dataloader, hqsam_extractor, device, config):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_l1_loss = 0
    total_iou_loss = 0
    ious = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images'].to(device)
            gt_bboxes = batch['gt_bboxes'].to(device)
            noisy_bboxes = batch['noisy_bboxes'].to(device)
            
            # 提取图像特征
            image_features_list = []
            for i in range(images.shape[0]):
                image_np = images[i].permute(1, 2, 0).cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)
                features = hqsam_extractor.extract_features(image_np)
                image_features_list.append(features)
            
            image_features = torch.cat(image_features_list, dim=0)
            
            # 前向传播
            refined_bboxes, _ = model.iterative_refine(
                image_features, noisy_bboxes, 
                (config['data']['image_size'], config['data']['image_size']),
                max_iter=config['refinement']['max_iter']
            )
            
            # 计算损失
            loss, l1_loss, iou_loss = compute_loss(
                refined_bboxes, gt_bboxes,
                l1_weight=config['loss']['l1_weight'],
                iou_weight=config['loss']['iou_weight']
            )
            
            total_loss += loss.item()
            total_l1_loss += l1_loss.item()
            total_iou_loss += iou_loss.item()
            
            # 计算IoU
            for i in range(refined_bboxes.shape[0]):
                iou = compute_bbox_iou(refined_bboxes[i], gt_bboxes[i])
                ious.append(iou.item())
    
    avg_loss = total_loss / len(dataloader)
    avg_l1_loss = total_l1_loss / len(dataloader)
    avg_iou_loss = total_iou_loss / len(dataloader)
    avg_iou = np.mean(ious)
    
    return avg_loss, avg_l1_loss, avg_iou_loss, avg_iou


def compute_bbox_iou(box1, box2):
    """计算两个bbox的IoU"""
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    iou = intersection / (union + 1e-6)
    return iou


def main():
    parser = argparse.ArgumentParser(description="Train Box Refinement Module")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.debug:
        config['debug']['enabled'] = True
        config['training']['epochs'] = 5
        config['training']['batch_size'] = 4
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
    print(f"Using device: {device}")
    
    # 创建输出目录
    for dir_name in ['save_dir', 'log_dir', 'checkpoint_dir', 'visualization_dir']:
        Path(config['output'][dir_name]).mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(config['output']['log_dir']) / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    # 创建数据集
    print("Loading datasets...")
    train_dataset = FungiDataset(
        data_root=config['data']['data_root'],
        split=config['data']['train_split'],
        image_size=config['data']['image_size'],
        data_subset=config['data']['data_subset'],
        augmentation=config['data']['augmentation']['enabled'],
        debug=config['debug']['enabled']
    )
    
    val_dataset = FungiDataset(
        data_root=config['data']['data_root'],
        split=config['data']['val_split'],
        image_size=config['data']['image_size'],
        data_subset=config['data']['data_subset'],
        augmentation=False,
        debug=config['debug']['enabled']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # 创建模型
    print("Creating model...")
    model = BoxRefinementModule(
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        max_offset=config['model']['max_offset']
    ).to(device)
    
    # 创建HQ-SAM特征提取器
    print("Loading HQ-SAM feature extractor...")
    hqsam_extractor = create_hqsam_extractor(
        checkpoint_path=config['hqsam']['checkpoint'],
        model_type=config['hqsam']['model_type'],
        device=device,
        use_mock=True  # 使用Mock版本进行测试
    )
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    if config['training']['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['epochs']
        )
    else:
        scheduler = None
    
    # 训练循环
    print("Starting training...")
    best_iou = 0
    
    for epoch in range(config['training']['epochs']):
        # 训练
        train_loss, train_l1, train_iou = train_one_epoch(
            model, train_loader, optimizer, hqsam_extractor, device, epoch, config
        )
        
        # 评估
        if epoch % config['evaluation']['val_freq'] == 0:
            val_loss, val_l1, val_iou, val_bbox_iou = evaluate(
                model, val_loader, hqsam_extractor, device, config
            )
            
            logging.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val IoU={val_bbox_iou:.4f}")
            
            # 保存最佳模型
            if val_bbox_iou > best_iou:
                best_iou = val_bbox_iou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_iou': best_iou,
                    'config': config
                }, Path(config['output']['checkpoint_dir']) / 'best_model.pth')
        else:
            logging.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}")
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
        
        # 定期保存
        if epoch % config['output']['save_freq'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, Path(config['output']['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pth')
    
    print(f"Training completed! Best IoU: {best_iou:.4f}")


if __name__ == '__main__':
    main()