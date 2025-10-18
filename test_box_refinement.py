#!/usr/bin/env python3
"""
测试 Box Refinement 模块的改进效果
"""

import torch
import numpy as np
import sys
import os

# 添加modules目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.box_refinement import BoxRefinementModule, box_iou_loss, compute_loss


def test_loss_functions():
    """测试损失函数"""
    print("Testing loss functions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    pred_boxes = torch.tensor([[0.2, 0.2, 0.8, 0.8], [0.1, 0.1, 0.9, 0.9]], device=device)
    target_boxes = torch.tensor([[0.25, 0.25, 0.75, 0.75], [0.15, 0.15, 0.85, 0.85]], device=device)
    
    # 测试IoU损失
    iou_loss = box_iou_loss(pred_boxes, target_boxes)
    print(f"IoU Loss: {iou_loss.item():.4f}")
    
    # 测试总损失
    total_loss, l1_loss, iou_loss = compute_loss(pred_boxes, target_boxes)
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"L1 Loss: {l1_loss.item():.4f}")
    print(f"IoU Loss: {iou_loss.item():.4f}")
    
    # 检查损失值是否合理
    assert total_loss.item() < 10.0, f"Loss too high: {total_loss.item()}"
    assert not torch.isnan(total_loss), "Loss is NaN"
    assert not torch.isinf(total_loss), "Loss is Inf"
    
    print("✅ Loss functions test passed!")


def test_model_forward():
    """测试模型前向传播"""
    print("Testing model forward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = BoxRefinementModule(hidden_dim=256, num_heads=8, max_offset=50).to(device)
    
    # 创建测试数据
    B = 2
    H, W = 300, 300
    image_embedding = torch.randn(B, 256, 64, 64, device=device)
    initial_bbox = torch.tensor([[0.2, 0.2, 0.8, 0.8], [0.1, 0.1, 0.9, 0.9]], device=device)
    img_shape = (H, W)
    
    # 测试单次前向传播
    offset = model(image_embedding, initial_bbox, img_shape)
    print(f"Offset shape: {offset.shape}")
    print(f"Offset values: {offset}")
    
    # 检查偏移量是否合理
    assert offset.shape == (B, 4), f"Wrong offset shape: {offset.shape}"
    assert torch.all(torch.abs(offset) <= 50), f"Offset too large: {offset.max()}"
    
    # 测试迭代精炼
    refined_bbox, history = model.iterative_refine(
        image_embedding, initial_bbox, img_shape, max_iter=3, stop_threshold=1.0
    )
    print(f"Refined bbox shape: {refined_bbox.shape}")
    print(f"Number of iterations: {len(history)}")
    
    # 检查精炼后的bbox是否合理
    assert refined_bbox.shape == (B, 4), f"Wrong refined bbox shape: {refined_bbox.shape}"
    assert torch.all(refined_bbox >= 0), "Refined bbox has negative values"
    assert torch.all(refined_bbox <= 1), "Refined bbox has values > 1"
    
    print("✅ Model forward pass test passed!")


def test_training_step():
    """测试训练步骤"""
    print("Testing training step...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型和优化器
    model = BoxRefinementModule(hidden_dim=256, num_heads=8, max_offset=50).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    # 创建测试数据
    B = 4
    H, W = 300, 300
    image_embedding = torch.randn(B, 256, 64, 64, device=device)
    noisy_bbox = torch.tensor([
        [0.2, 0.2, 0.8, 0.8],
        [0.1, 0.1, 0.9, 0.9],
        [0.3, 0.3, 0.7, 0.7],
        [0.15, 0.15, 0.85, 0.85]
    ], device=device)
    gt_bbox = torch.tensor([
        [0.25, 0.25, 0.75, 0.75],
        [0.15, 0.15, 0.85, 0.85],
        [0.35, 0.35, 0.65, 0.65],
        [0.2, 0.2, 0.8, 0.8]
    ], device=device)
    img_shape = (H, W)
    
    # 训练步骤
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    refined_bbox, history = model.iterative_refine(
        image_embedding, noisy_bbox, img_shape, max_iter=3, stop_threshold=1.0
    )
    
    # 计算损失
    loss, l1_loss, iou_loss = compute_loss(refined_bbox, gt_bbox)
    
    print(f"Training Loss: {loss.item():.4f}")
    print(f"L1 Loss: {l1_loss.item():.4f}")
    print(f"IoU Loss: {iou_loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    total_grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"Gradient norm: {total_grad_norm:.4f}")
    
    # 检查梯度是否合理
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    assert total_grad_norm < 100, f"Gradient norm too large: {total_grad_norm}"
    
    # 更新参数
    optimizer.step()
    
    print("✅ Training step test passed!")


def test_performance():
    """测试性能"""
    print("Testing performance...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = BoxRefinementModule(hidden_dim=256, num_heads=8, max_offset=50).to(device)
    model.eval()
    
    # 创建测试数据
    B = 8
    H, W = 300, 300
    image_embedding = torch.randn(B, 256, 64, 64, device=device)
    initial_bbox = torch.rand(B, 4, device=device)
    img_shape = (H, W)
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model.iterative_refine(image_embedding, initial_bbox, img_shape)
    
    # 测试性能
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            _ = model.iterative_refine(image_embedding, initial_bbox, img_shape)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"Average inference time: {avg_time*1000:.2f}ms per batch")
    
    # 检查性能是否合理
    assert avg_time < 1.0, f"Inference too slow: {avg_time:.2f}s"
    
    print("✅ Performance test passed!")


def main():
    """主测试函数"""
    print("🧪 Testing Box Refinement Module Improvements")
    print("=" * 50)
    
    try:
        test_loss_functions()
        print()
        
        test_model_forward()
        print()
        
        test_training_step()
        print()
        
        test_performance()
        print()
        
        print("🎉 All tests passed! The improvements are working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    import time
    main()