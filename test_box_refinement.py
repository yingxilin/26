"""
测试脚本: 在验证集上评估 Box Refinement 的效果
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
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加modules目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.box_refinement import BoxRefinementModule, visualize_refinement
from modules.hqsam_feature_extractor import create_hqsam_extractor
from train_box_refiner import FungiDataset, collate_fn, compute_bbox_iou


def evaluate_model(model, dataloader, hqsam_extractor, device, config, save_results=True):
    """
    评估模型性能
    
    Args:
        model: 训练好的BoxRefinementModule
        dataloader: 验证集数据加载器
        hqsam_extractor: HQ-SAM特征提取器
        device: 设备
        config: 配置字典
        save_results: 是否保存结果
    
    Returns:
        results: 评估结果字典
    """
    model.eval()
    
    # 统计指标
    metrics = {
        'bbox_iou': [],
        'offset_magnitude': [],
        'iteration_count': [],
        'l1_error': [],
        'improvement_iou': []  # 相对于noisy bbox的IoU提升
    }
    
    # 可视化样本
    visualization_samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
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
            
            image_features = torch.cat(image_features_list, dim=0)
            
            # 前向传播 - 获取迭代历史
            refined_bboxes, history = model.iterative_refine(
                image_features, noisy_bboxes,
                (config['data']['image_size'], config['data']['image_size']),
                max_iter=config['refinement']['max_iter'],
                stop_threshold=config['refinement']['stop_threshold']
            )
            
            # 计算各种指标
            for i in range(images.shape[0]):
                # IoU
                bbox_iou = compute_bbox_iou(refined_bboxes[i], gt_bboxes[i])
                metrics['bbox_iou'].append(bbox_iou.item())
                
                # 偏移量大小
                offset = refined_bboxes[i] - noisy_bboxes[i]
                offset_magnitude = torch.norm(offset).item()
                metrics['offset_magnitude'].append(offset_magnitude)
                
                # 迭代次数
                iteration_count = len(history) - 1  # 减去初始状态
                metrics['iteration_count'].append(iteration_count)
                
                # L1误差
                l1_error = torch.nn.L1Loss()(refined_bboxes[i], gt_bboxes[i]).item()
                metrics['l1_error'].append(l1_error)
                
                # 相对于noisy bbox的IoU提升
                noisy_iou = compute_bbox_iou(noisy_bboxes[i], gt_bboxes[i])
                improvement = bbox_iou.item() - noisy_iou.item()
                metrics['improvement_iou'].append(improvement)
            
            # 收集可视化样本（前几个batch）
            if batch_idx < 3:  # 只收集前3个batch用于可视化
                for i in range(min(2, images.shape[0])):  # 每个batch最多2个样本
                    sample = {
                        'image': images[i].permute(1, 2, 0).cpu().numpy(),
                        'gt_bbox': gt_bboxes[i].cpu().numpy(),
                        'noisy_bbox': noisy_bboxes[i].cpu().numpy(),
                        'refined_bbox': refined_bboxes[i].cpu().numpy(),
                        'history': [h[i].cpu().numpy() for h in history],
                        'image_path': image_paths[i]
                    }
                    visualization_samples.append(sample)
    
    # 计算统计结果
    results = {}
    for metric_name, values in metrics.items():
        results[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # 打印结果
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of samples: {len(metrics['bbox_iou'])}")
    print(f"Bbox IoU: {results['bbox_iou']['mean']:.4f} ± {results['bbox_iou']['std']:.4f}")
    print(f"Offset Magnitude: {results['offset_magnitude']['mean']:.2f} ± {results['offset_magnitude']['std']:.2f} pixels")
    print(f"Iteration Count: {results['iteration_count']['mean']:.2f} ± {results['iteration_count']['std']:.2f}")
    print(f"L1 Error: {results['l1_error']['mean']:.2f} ± {results['l1_error']['std']:.2f} pixels")
    print(f"IoU Improvement: {results['improvement_iou']['mean']:.4f} ± {results['improvement_iou']['std']:.4f}")
    print("="*60)
    
    # 保存结果
    if save_results:
        output_dir = Path(config['output']['save_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        np.save(output_dir / 'bbox_ious.npy', metrics['bbox_iou'])
        np.save(output_dir / 'offset_magnitudes.npy', metrics['offset_magnitude'])
        np.save(output_dir / 'iteration_counts.npy', metrics['iteration_count'])
        np.save(output_dir / 'l1_errors.npy', metrics['l1_error'])
        np.save(output_dir / 'improvements.npy', metrics['improvement_iou'])
        
        # 保存统计结果
        with open(output_dir / 'evaluation_results.txt', 'w') as f:
            f.write("Box Refinement Evaluation Results\n")
            f.write("="*50 + "\n")
            f.write(f"Number of samples: {len(metrics['bbox_iou'])}\n")
            f.write(f"Bbox IoU: {results['bbox_iou']['mean']:.4f} ± {results['bbox_iou']['std']:.4f}\n")
            f.write(f"Offset Magnitude: {results['offset_magnitude']['mean']:.2f} ± {results['offset_magnitude']['std']:.2f} pixels\n")
            f.write(f"Iteration Count: {results['iteration_count']['mean']:.2f} ± {results['iteration_count']['std']:.2f}\n")
            f.write(f"L1 Error: {results['l1_error']['mean']:.2f} ± {results['l1_error']['std']:.2f} pixels\n")
            f.write(f"IoU Improvement: {results['improvement_iou']['mean']:.4f} ± {results['improvement_iou']['std']:.4f}\n")
        
        print(f"Results saved to: {output_dir}")
    
    return results, visualization_samples


def visualize_samples(visualization_samples, config, num_samples=10):
    """可视化精炼过程"""
    vis_dir = Path(config['output']['visualization_dir'])
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = min(num_samples, len(visualization_samples))
    
    for i in range(num_samples):
        sample = visualization_samples[i]
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        image = (sample['image'] * 255).astype(np.uint8)
        
        # GT bbox
        axes[0].imshow(image)
        gt_rect = plt.Rectangle((sample['gt_bbox'][0], sample['gt_bbox'][1]), 
                               sample['gt_bbox'][2]-sample['gt_bbox'][0], 
                               sample['gt_bbox'][3]-sample['gt_bbox'][1],
                               linewidth=3, edgecolor='green', facecolor='none', label='GT')
        axes[0].add_patch(gt_rect)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        # Noisy bbox
        axes[1].imshow(image)
        noisy_rect = plt.Rectangle((sample['noisy_bbox'][0], sample['noisy_bbox'][1]), 
                                  sample['noisy_bbox'][2]-sample['noisy_bbox'][0], 
                                  sample['noisy_bbox'][3]-sample['noisy_bbox'][1],
                                  linewidth=3, edgecolor='red', facecolor='none', label='Noisy')
        axes[1].add_patch(noisy_rect)
        axes[1].set_title('Noisy (YOLO)')
        axes[1].axis('off')
        
        # Refined bbox + 迭代过程
        axes[2].imshow(image)
        
        # 绘制迭代过程
        colors = ['red', 'orange', 'yellow', 'blue', 'green']
        for j, bbox in enumerate(sample['history']):
            if j >= len(colors):
                color = 'purple'
            else:
                color = colors[j]
            
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                               linewidth=2, edgecolor=color, facecolor='none',
                               label=f'Iter {j}' if j < len(colors) else f'Iter {j}')
            axes[2].add_patch(rect)
            
            # 添加箭头显示移动方向
            if j < len(sample['history']) - 1:
                next_bbox = sample['history'][j + 1]
                curr_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                next_center = [(next_bbox[0] + next_bbox[2]) / 2, (next_bbox[1] + next_bbox[3]) / 2]
                
                axes[2].annotate('', xy=next_center, xytext=curr_center,
                               arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        
        # 最终精炼结果
        refined_rect = plt.Rectangle((sample['refined_bbox'][0], sample['refined_bbox'][1]), 
                                    sample['refined_bbox'][2]-sample['refined_bbox'][0], 
                                    sample['refined_bbox'][3]-sample['refined_bbox'][1],
                                    linewidth=3, edgecolor='purple', facecolor='none', 
                                    linestyle='--', label='Final')
        axes[2].add_patch(refined_rect)
        axes[2].set_title('Refinement Process')
        axes[2].axis('off')
        
        plt.suptitle(f'Sample {i+1}: {Path(sample["image_path"]).name}')
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(vis_dir / f'sample_{i+1:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Visualization saved to: {vis_dir}")


def plot_metrics(results, config):
    """绘制评估指标图表"""
    output_dir = Path(config['output']['save_dir'])
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # IoU分布
    axes[0].hist(results['bbox_iou']['mean'], bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(results['bbox_iou']['mean'], color='r', linestyle='--', linewidth=2)
    axes[0].set_title('Bbox IoU Distribution')
    axes[0].set_xlabel('IoU')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # 偏移量分布
    axes[1].hist(results['offset_magnitude']['mean'], bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(results['offset_magnitude']['mean'], color='r', linestyle='--', linewidth=2)
    axes[1].set_title('Offset Magnitude Distribution')
    axes[1].set_xlabel('Offset (pixels)')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    # 迭代次数分布
    axes[2].hist(results['iteration_count']['mean'], bins=20, alpha=0.7, edgecolor='black')
    axes[2].axvline(results['iteration_count']['mean'], color='r', linestyle='--', linewidth=2)
    axes[2].set_title('Iteration Count Distribution')
    axes[2].set_xlabel('Iterations')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, alpha=0.3)
    
    # L1误差分布
    axes[3].hist(results['l1_error']['mean'], bins=50, alpha=0.7, edgecolor='black')
    axes[3].axvline(results['l1_error']['mean'], color='r', linestyle='--', linewidth=2)
    axes[3].set_title('L1 Error Distribution')
    axes[3].set_xlabel('L1 Error (pixels)')
    axes[3].set_ylabel('Frequency')
    axes[3].grid(True, alpha=0.3)
    
    # IoU提升分布
    axes[4].hist(results['improvement_iou']['mean'], bins=50, alpha=0.7, edgecolor='black')
    axes[4].axvline(results['improvement_iou']['mean'], color='r', linestyle='--', linewidth=2)
    axes[4].set_title('IoU Improvement Distribution')
    axes[4].set_xlabel('IoU Improvement')
    axes[4].set_ylabel('Frequency')
    axes[4].grid(True, alpha=0.3)
    
    # 隐藏最后一个子图
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics plots saved to: {output_dir / 'evaluation_metrics.png'}")


def main():
    parser = argparse.ArgumentParser(description="Test Box Refinement Module")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", help="Output directory for results")
    parser.add_argument("--num_vis_samples", type=int, default=10, help="Number of visualization samples")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.debug:
        config['debug']['enabled'] = True
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
    print(f"Using device: {device}")
    
    # 创建输出目录
    if args.output_dir:
        config['output']['save_dir'] = args.output_dir
        config['output']['visualization_dir'] = os.path.join(args.output_dir, 'visualizations')
    
    for dir_name in ['save_dir', 'visualization_dir']:
        Path(config['output'][dir_name]).mkdir(parents=True, exist_ok=True)
    
    # 创建数据集
    print("Loading validation dataset...")
    val_dataset = FungiDataset(
        data_root=config['data']['data_root'],
        split=config['data']['val_split'],
        image_size=config['data']['image_size'],
        data_subset=config['data']['data_subset'],
        augmentation=False,
        debug=config['debug']['enabled']
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # 创建模型
    print("Loading model...")
    model = BoxRefinementModule(
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        max_offset=config['model']['max_offset']
    ).to(device)
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # 创建HQ-SAM特征提取器
    print("Loading HQ-SAM feature extractor...")
    hqsam_extractor = create_hqsam_extractor(
        checkpoint_path=config['hqsam']['checkpoint'],
        model_type=config['hqsam']['model_type'],
        device=device,
        use_mock=True  # 使用Mock版本进行测试
    )
    
    # 评估模型
    print("Evaluating model...")
    results, visualization_samples = evaluate_model(
        model, val_loader, hqsam_extractor, device, config, save_results=True
    )
    
    # 可视化样本
    print("Creating visualizations...")
    visualize_samples(visualization_samples, config, args.num_vis_samples)
    
    # 绘制指标图表
    print("Plotting metrics...")
    plot_metrics(results, config)
    
    print("Evaluation completed!")


if __name__ == '__main__':
    main()