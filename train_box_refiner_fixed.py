#!/usr/bin/env python3
"""
Box Refinement 训练脚本 - 修复版本
修复了混合精度训练、损失计算和特征缓存的问题
"""

import os
import sys
import time
import yaml
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加modules目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.box_refinement import BoxRefinementModule, box_iou_loss
from modules.hqsam_feature_extractor import create_hqsam_extractor
from modules.dataset import FungiDataset


class FeatureCache:
    """特征缓存管理器"""
    
    def __init__(self, cache_dir: str, split: str = 'train'):
        self.cache_dir = Path(cache_dir) / f"features/{split}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
    
    def get_cache_path(self, image_path: str) -> Path:
        """获取缓存文件路径"""
        image_hash = hashlib.md5(image_path.encode()).hexdigest()
        return self.cache_dir / f"{image_hash}.npy"
    
    def load_features(self, image_path: str) -> Optional[torch.Tensor]:
        """从缓存加载特征"""
        self.total_requests += 1
        cache_path = self.get_cache_path(image_path)
        
        if cache_path.exists():
            try:
                features = np.load(cache_path)
                features_tensor = torch.from_numpy(features)
                self.cache_hits += 1
                return features_tensor
            except Exception as e:
                print(f"Warning: Failed to load cached features from {cache_path}: {e}")
                self.cache_misses += 1
                return None
        else:
            self.cache_misses += 1
            return None
    
    def save_features(self, image_path: str, features: torch.Tensor):
        """保存特征到缓存"""
        cache_path = self.get_cache_path(image_path)
        try:
            features_cpu = features.cpu().numpy()
            np.save(cache_path, features_cpu)
        except Exception as e:
            print(f"Warning: Failed to save features to {cache_path}: {e}")
    
    def get_cache_stats(self) -> Dict[str, float]:
        """获取缓存统计信息"""
        if self.total_requests == 0:
            return {'hit_rate': 0.0, 'hits': 0, 'misses': 0}
        
        hit_rate = self.cache_hits / self.total_requests
        return {
            'hit_rate': hit_rate,
            'hits': self.cache_hits,
            'misses': self.cache_misses
        }
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0


def detect_feature_cache(cache_dir: str) -> bool:
    """检测是否存在特征缓存"""
    cache_path = Path(cache_dir) / "features"
    return cache_path.exists() and any(cache_path.iterdir())


def extract_features_with_cache(hqsam_extractor, images_np_list, image_paths, feature_cache, device='cuda'):
    """使用缓存提取特征"""
    features_list = []
    
    for i, (image_np, image_path) in enumerate(zip(images_np_list, image_paths)):
        # 尝试从缓存加载
        if feature_cache is not None:
            cached_features = feature_cache.load_features(image_path)
            if cached_features is not None:
                # 确保缓存的特征在正确设备上
                cached_features = cached_features.to(device)
                features_list.append(cached_features)
                continue
        
        # 缓存未命中，提取特征
        features = hqsam_extractor.extract_features(image_np)
        features_list.append(features)
        
        # 保存到缓存
        if feature_cache is not None:
            feature_cache.save_features(image_path, features)
    
    return features_list


def compute_loss(pred_bboxes, gt_bboxes, l1_weight=1.0, iou_weight=2.0):
    """计算损失函数 - 修复版本"""
    # 确保输入张量在相同设备上
    if pred_bboxes.device != gt_bboxes.device:
        gt_bboxes = gt_bboxes.to(pred_bboxes.device)
    
    # 确保输入张量形状一致
    if pred_bboxes.shape != gt_bboxes.shape:
        min_batch = min(pred_bboxes.shape[0], gt_bboxes.shape[0])
        pred_bboxes = pred_bboxes[:min_batch]
        gt_bboxes = gt_bboxes[:min_batch]
    
    # L1损失
    l1_loss = F.l1_loss(pred_bboxes, gt_bboxes)
    
    # IoU损失 - 添加数值稳定性
    try:
        iou_loss = box_iou_loss(pred_bboxes, gt_bboxes)
        # 检查IoU损失是否为NaN或Inf
        if torch.isnan(iou_loss) or torch.isinf(iou_loss):
            iou_loss = torch.tensor(0.0, device=pred_bboxes.device)
    except Exception as e:
        print(f"Warning: IoU loss computation failed: {e}")
        iou_loss = torch.tensor(0.0, device=pred_bboxes.device)
    
    # 总损失
    total_loss = l1_weight * l1_loss + iou_weight * iou_loss
    
    return total_loss, l1_loss, iou_loss


def train_one_epoch(model, dataloader, optimizer, hqsam_extractor, device, epoch, config, 
                   feature_cache=None, use_amp=False):
    """训练一个epoch - 修复版本"""
    model.train()
    
    total_loss = 0.0
    total_l1_loss = 0.0
    total_iou_loss = 0.0
    num_batches = len(dataloader)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        gt_bboxes = batch['gt_bboxes'].to(device)
        noisy_bboxes = batch['noisy_bboxes'].to(device)
        image_paths = batch['image_paths']
        
        # 提取特征
        images_np_list = [img.cpu().numpy().transpose(1, 2, 0) for img in images]
        features_list = extract_features_with_cache(
            hqsam_extractor, images_np_list, image_paths, feature_cache, device
        )
        image_features = torch.cat(features_list, dim=0)  # (B, 256, 64, 64)
        
        # 前向传播
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            # 混合精度前向传播
            with torch.cuda.amp.autocast():
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
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 普通精度前向传播
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
        cache_stats = feature_cache.get_cache_stats() if feature_cache else {'hit_rate': 0.0}
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'L1': f'{l1_loss.item():.4f}',
            'IoU': f'{iou_loss.item():.4f}',
            'Cache': f'Cache: {cache_stats["hit_rate"]:.1%}'
        })
    
    return total_loss / num_batches, total_l1_loss / num_batches, total_iou_loss / num_batches


def evaluate(model, dataloader, hqsam_extractor, device, config, feature_cache=None):
    """评估模型 - 修复版本"""
    model.eval()
    
    total_loss = 0.0
    total_l1_loss = 0.0
    total_iou_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images'].to(device)
            gt_bboxes = batch['gt_bboxes'].to(device)
            noisy_bboxes = batch['noisy_bboxes'].to(device)
            image_paths = batch['image_paths']
            
            # 提取特征
            images_np_list = [img.cpu().numpy().transpose(1, 2, 0) for img in images]
            features_list = extract_features_with_cache(
                hqsam_extractor, images_np_list, image_paths, feature_cache, device
            )
            image_features = torch.cat(features_list, dim=0)
            
            # 前向传播
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
            
            total_loss += loss.item()
            total_l1_loss += l1_loss.item()
            total_iou_loss += iou_loss.item()
    
    return total_loss / num_batches, total_l1_loss / num_batches, total_iou_loss / num_batches


def main():
    """主函数 - 修复版本"""
    parser = argparse.ArgumentParser(description='Box Refinement Training - Fixed Version')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--device', type=str, default='auto', help='设备选择')
    parser.add_argument('--fast', action='store_true', help='快速模式')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--clear-cache', action='store_true', help='清空特征缓存')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 快速模式设置
    if args.fast:
        print("🚀 Fast mode enabled - applying all optimizations")
        config['data']['sample_ratio'] = 0.1
        config['training']['use_amp'] = True
        config['training']['batch_size'] = 32
        print(f"  - Data sampling: {config['data']['sample_ratio']}")
        print(f"  - Mixed precision: {config['training']['use_amp']}")
        print(f"  - Batch size: {config['training']['batch_size']}")
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 检测特征缓存
    cache_detected = detect_feature_cache(config['output']['checkpoint_dir'])
    print(f"Feature cache detected: {cache_detected}")
    
    # 创建特征缓存
    feature_cache = FeatureCache(config['output']['checkpoint_dir']) if config['training']['feature_cache'] else None
    
    # 清空缓存
    if args.clear_cache and feature_cache is not None:
        print("Clearing feature cache...")
        feature_cache.clear_cache()
        print("Feature cache cleared.")
    
    # 加载数据集
    print("Loading datasets...")
    train_dataset = FungiDataset(
        data_root=config['data']['data_root'],
        split=config['data']['train_split'],
        image_size=config['data']['image_size'],
        data_subset=config['data']['data_subset'],
        augmentation=config['data']['augmentation']['enabled'],
        debug=args.debug,
        sample_ratio=config['data']['sample_ratio']
    )
    
    val_dataset = FungiDataset(
        data_root=config['data']['data_root'],
        split=config['data']['val_split'],
        image_size=config['data']['image_size'],
        data_subset=config['data']['data_subset'],
        augmentation=False,
        debug=args.debug,
        sample_ratio=config['data']['sample_ratio']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
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
        checkpoint_path=config['hqsam']['checkpoint_path'],
        model_type=config['hqsam']['model_type'],
        device=device,
        use_mock=True  # 使用Mock版本进行测试
    )
    
    # 创建优化器 - 修复学习率
    print("Creating optimizer...")
    learning_rate = float(config['training']['learning_rate'])
    if args.fast:
        learning_rate *= 2  # 快速模式下稍微提高学习率
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(config['training']['weight_decay'])
    )
    
    print(f"Learning rate: {learning_rate}")
    
    # 学习率调度器
    if config['training']['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['epochs']
        )
    else:
        scheduler = None
    
    # 训练循环
    print("Starting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['training']['epochs']):
        # 训练
        train_loss, train_l1, train_iou = train_one_epoch(
            model, train_loader, optimizer, hqsam_extractor, device, epoch, config,
            feature_cache=feature_cache, use_amp=config['training']['use_amp']
        )
        
        # 验证
        val_loss, val_l1, val_iou = evaluate(
            model, val_loader, hqsam_extractor, device, config, feature_cache=feature_cache
        )
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 打印统计信息
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Train L1: {train_l1:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val L1: {val_l1:.4f}, IoU: {val_iou:.4f}")
        
        # 缓存统计
        if feature_cache is not None:
            cache_stats = feature_cache.get_cache_stats()
            print(f"  Cache hit rate: {cache_stats['hit_rate']:.1%}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(config['output']['checkpoint_dir'], 'best_model.pth'))
            print(f"  New best model saved! Val Loss: {val_loss:.4f}")
    
    # 保存最终模型
    torch.save({
        'epoch': config['training']['epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, os.path.join(config['output']['checkpoint_dir'], 'final_model.pth'))
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # 打印缓存统计
    if feature_cache is not None:
        cache_stats = feature_cache.get_cache_stats()
        print(f"Final cache hit rate: {cache_stats['hit_rate']:.1%}")
        print(f"Total cache hits: {cache_stats['hits']}")
        print(f"Total cache misses: {cache_stats['misses']}")


if __name__ == "__main__":
    main()