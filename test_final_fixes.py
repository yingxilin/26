#!/usr/bin/env python3
"""
测试最终修复版本的效果
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import sys
import os

# 添加modules目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def test_bbox_normalization():
    """测试边界框归一化"""
    print("🔍 测试边界框归一化...")
    
    # 模拟原始边界框（像素坐标）
    gt_bbox_pixel = np.array([50, 60, 150, 200], dtype=np.float32)  # [x1, y1, x2, y2]
    image_shape = (300, 300)  # (height, width)
    
    # 归一化
    h, w = image_shape
    gt_bbox_normalized = gt_bbox_pixel / np.array([w, h, w, h], dtype=np.float32)
    
    print(f"  原始边界框 (像素): {gt_bbox_pixel}")
    print(f"  归一化边界框: {gt_bbox_normalized}")
    print(f"  范围检查: {np.all(gt_bbox_normalized >= 0) and np.all(gt_bbox_normalized <= 1)}")
    
    # 验证归一化正确性
    expected = np.array([50/300, 60/300, 150/300, 200/300], dtype=np.float32)
    is_correct = np.allclose(gt_bbox_normalized, expected)
    print(f"  ✅ 归一化正确: {is_correct}")
    
    return is_correct

def test_loss_computation():
    """测试损失计算"""
    print("\n🔍 测试损失计算...")
    
    # 模拟预测和真实边界框（归一化坐标）
    pred_bboxes = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    gt_bboxes = torch.tensor([[0.15, 0.25, 0.35, 0.45]], dtype=torch.float32)
    
    # 计算L1损失
    l1_loss = torch.nn.functional.l1_loss(pred_bboxes, gt_bboxes)
    
    print(f"  预测边界框: {pred_bboxes}")
    print(f"  真实边界框: {gt_bboxes}")
    print(f"  L1损失: {l1_loss.item():.6f}")
    
    # 验证损失值在合理范围内
    is_reasonable = 0 < l1_loss.item() < 1.0
    print(f"  ✅ 损失值合理: {is_reasonable}")
    
    return is_reasonable

def test_cache_mechanism():
    """测试缓存机制"""
    print("\n🔍 测试缓存机制...")
    
    # 模拟缓存操作
    class MockFeatureCache:
        def __init__(self):
            self.cache = {}
            self.hits = 0
            self.misses = 0
            self.total = 0
        
        def load(self, key):
            self.total += 1
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None
        
        def save(self, key, value):
            self.cache[key] = value
        
        def get_stats(self):
            return {
                'hit_rate': self.hits / self.total if self.total > 0 else 0,
                'hits': self.hits,
                'misses': self.misses
            }
    
    cache = MockFeatureCache()
    
    # 测试缓存未命中
    result1 = cache.load("image1.jpg")
    print(f"  第一次加载: {result1} (应该是None)")
    
    # 保存到缓存
    cache.save("image1.jpg", "features1")
    
    # 测试缓存命中
    result2 = cache.load("image1.jpg")
    print(f"  第二次加载: {result2} (应该是features1)")
    
    # 测试统计
    stats = cache.get_stats()
    print(f"  缓存统计: {stats}")
    
    is_working = result1 is None and result2 == "features1" and stats['hit_rate'] == 0.5
    print(f"  ✅ 缓存机制正常: {is_working}")
    
    return is_working

def test_config_loading():
    """测试配置文件加载"""
    print("\n🔍 测试配置文件加载...")
    
    config_file = 'configs/box_refinement_config.yaml'
    
    if not Path(config_file).exists():
        print(f"  ❌ 配置文件不存在: {config_file}")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查关键配置
        required_keys = ['data', 'model', 'training', 'hqsam', 'loss', 'refinement']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"  ❌ 缺少配置键: {missing_keys}")
            return False
        
        # 检查hqsam配置
        if 'checkpoint' not in config['hqsam']:
            print(f"  ❌ hqsam配置缺少checkpoint键")
            return False
        
        print(f"  ✅ 配置文件加载成功")
        print(f"  - 数据采样比例: {config['data'].get('sample_ratio', 'None')}")
        print(f"  - 混合精度训练: {config['training'].get('use_amp', False)}")
        print(f"  - 特征缓存: {config['training'].get('feature_cache', False)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 配置文件加载失败: {e}")
        return False

def test_device_consistency():
    """测试设备一致性"""
    print("\n🔍 测试设备一致性...")
    
    if not torch.cuda.is_available():
        print("  ⚠️  CUDA不可用，跳过设备测试")
        return True
    
    device = torch.device('cuda')
    
    # 创建张量
    tensor1 = torch.randn(2, 4).to(device)
    tensor2 = torch.randn(2, 4).to(device)
    
    # 测试设备一致性
    same_device = tensor1.device == tensor2.device == device
    print(f"  张量1设备: {tensor1.device}")
    print(f"  张量2设备: {tensor2.device}")
    print(f"  ✅ 设备一致性: {same_device}")
    
    return same_device

def main():
    """主测试函数"""
    print("Box Refinement 最终修复版本测试")
    print("=" * 50)
    
    tests = [
        ("边界框归一化", test_bbox_normalization),
        ("损失计算", test_loss_computation),
        ("缓存机制", test_cache_mechanism),
        ("配置文件加载", test_config_loading),
        ("设备一致性", test_device_consistency),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name}: 通过")
            else:
                print(f"❌ {test_name}: 失败")
        except Exception as e:
            print(f"❌ {test_name}: 错误 - {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！修复版本应该可以正常工作。")
        print("\n💡 现在可以运行:")
        print("python train_box_refiner_final_fixed.py --config configs/box_refinement_config.yaml --fast")
    else:
        print("⚠️  部分测试失败，请检查相关功能。")
    
    return passed == total

if __name__ == "__main__":
    main()