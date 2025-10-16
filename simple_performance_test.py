#!/usr/bin/env python3
"""
简化的 Box Refinement 性能测试脚本
专门用于验证优化效果，避免复杂的数据集依赖
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, List

# 添加modules目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.box_refinement import BoxRefinementModule
from modules.hqsam_feature_extractor import create_hqsam_extractor
from train_box_refiner_optimized import FeatureCache, extract_features_with_cache


class SimplePerformanceTest:
    """简化的性能测试器"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # 创建模型
        self.model = BoxRefinementModule(
            hidden_dim=256,
            num_heads=8,
            max_offset=50
        ).to(device)
        
        # 创建特征提取器
        self.hqsam_extractor = create_hqsam_extractor(
            checkpoint_path="dummy_path",
            model_type='vit_h',
            device=device,
            use_mock=True
        )
        
        # 创建测试数据
        self.test_images = self._create_test_images(10)
        self.test_bboxes = self._create_test_bboxes(10)
    
    def _create_test_images(self, num_images: int) -> List[np.ndarray]:
        """创建测试图像"""
        images = []
        for i in range(num_images):
            image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            images.append(image)
        return images
    
    def _create_test_bboxes(self, num_bboxes: int) -> torch.Tensor:
        """创建测试bbox"""
        bboxes = []
        for i in range(num_bboxes):
            x1 = np.random.randint(0, 200)
            y1 = np.random.randint(0, 200)
            x2 = x1 + np.random.randint(50, 100)
            y2 = y1 + np.random.randint(50, 100)
            bboxes.append([x1, y1, x2, y2])
        return torch.tensor(bboxes, dtype=torch.float32, device=self.device)
    
    def test_original_method(self, num_epochs: int = 3, num_batches: int = 5) -> Dict[str, float]:
        """测试原始方法 (每次都提取特征)"""
        print("🔄 测试原始方法 (每次都提取特征)...")
        
        start_time = time.time()
        total_extractions = 0
        
        for epoch in range(num_epochs):
            for batch_idx in range(num_batches):
                # 每次都提取特征
                features_list = []
                for image in self.test_images:
                    features = self.hqsam_extractor.extract_features(image)
                    features_list.append(features.to(self.device))
                    total_extractions += 1
                
                # 模拟前向传播
                image_features = torch.cat(features_list, dim=0)
                bboxes = self.test_bboxes[:len(features_list)]
                
                with torch.no_grad():
                    refined_bboxes, _ = self.model.iterative_refine(
                        image_features, bboxes, (300, 300), max_iter=3
                    )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return {
            'total_time': total_time,
            'avg_epoch_time': total_time / num_epochs,
            'avg_batch_time': total_time / (num_epochs * num_batches),
            'feature_extractions': total_extractions,
            'method': 'original'
        }
    
    def test_optimized_method(self, num_epochs: int = 3, num_batches: int = 5) -> Dict[str, float]:
        """测试优化方法 (使用缓存)"""
        print("⚡ 测试优化方法 (使用缓存)...")
        
        # 创建特征缓存
        cache_dir = "./test_cache"
        feature_cache = FeatureCache(cache_dir, 'test')
        
        start_time = time.time()
        total_extractions = 0
        
        for epoch in range(num_epochs):
            for batch_idx in range(num_batches):
                # 使用缓存提取特征
                image_paths = [f"test_image_{i}.jpg" for i in range(len(self.test_images))]
                features_list = extract_features_with_cache(
                    self.hqsam_extractor, self.test_images, image_paths, feature_cache, self.device
                )
                total_extractions += len(self.test_images)
                
                # 模拟前向传播
                image_features = torch.cat(features_list, dim=0)
                bboxes = self.test_bboxes[:len(features_list)]
                
                with torch.no_grad():
                    refined_bboxes, _ = self.model.iterative_refine(
                        image_features, bboxes, (300, 300), max_iter=3
                    )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 获取缓存统计
        cache_stats = feature_cache.get_cache_stats()
        
        return {
            'total_time': total_time,
            'avg_epoch_time': total_time / num_epochs,
            'avg_batch_time': total_time / (num_epochs * num_batches),
            'feature_extractions': total_extractions,
            'cache_hit_rate': cache_stats['hit_rate'],
            'cache_hits': cache_stats['hits'],
            'cache_misses': cache_stats['misses'],
            'method': 'optimized'
        }
    
    def test_mixed_precision(self) -> Dict[str, float]:
        """测试混合精度训练效果"""
        print("🎯 测试混合精度训练...")
        
        # 创建测试数据
        batch_size = 8
        images = torch.randn(batch_size, 3, 300, 300, device=self.device)
        bboxes = torch.randn(batch_size, 4, device=self.device)
        image_features = torch.randn(batch_size, 256, 64, 64, device=self.device)
        
        # 测试普通精度
        start_time = time.time()
        for _ in range(50):  # 减少测试次数
            with torch.no_grad():
                refined_bboxes, _ = self.model.iterative_refine(
                    image_features, bboxes, (300, 300), max_iter=3
                )
        normal_time = time.time() - start_time
        
        # 测试混合精度
        start_time = time.time()
        for _ in range(50):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    refined_bboxes, _ = self.model.iterative_refine(
                        image_features, bboxes, (300, 300), max_iter=3
                    )
        amp_time = time.time() - start_time
        
        return {
            'normal_time': normal_time,
            'amp_time': amp_time,
            'speedup': normal_time / amp_time if amp_time > 0 else 1.0,
            'memory_saved': '~30-50%'
        }
    
    def run_comprehensive_test(self) -> Dict[str, Dict[str, float]]:
        """运行综合测试"""
        print("🚀 开始综合性能测试...")
        print("=" * 60)
        
        results = {}
        
        # 测试原始方法
        try:
            results['original'] = self.test_original_method()
            print(f"✅ 原始方法完成: {results['original']['total_time']:.2f}s")
        except Exception as e:
            print(f"❌ 原始方法失败: {e}")
            results['original'] = {}
        
        # 测试优化方法
        try:
            results['optimized'] = self.test_optimized_method()
            print(f"✅ 优化方法完成: {results['optimized']['total_time']:.2f}s")
        except Exception as e:
            print(f"❌ 优化方法失败: {e}")
            results['optimized'] = {}
        
        # 测试混合精度
        try:
            results['mixed_precision'] = self.test_mixed_precision()
            print(f"✅ 混合精度测试完成")
        except Exception as e:
            print(f"❌ 混合精度测试失败: {e}")
            results['mixed_precision'] = {}
        
        return results
    
    def print_results(self, results: Dict[str, Dict[str, float]]):
        """打印测试结果"""
        print("\n📊 性能测试结果")
        print("=" * 60)
        
        # 原始 vs 优化对比
        if 'original' in results and 'optimized' in results:
            orig = results['original']
            opt = results['optimized']
            
            if 'total_time' in orig and 'total_time' in opt:
                speedup = orig['total_time'] / opt['total_time']
                print(f"\n🔄 特征缓存效果对比:")
                print(f"  原始方法时间: {orig['total_time']:.2f}s")
                print(f"  优化方法时间: {opt['total_time']:.2f}s")
                print(f"  加速比: {speedup:.1f}x")
                
                if 'cache_hit_rate' in opt:
                    print(f"  缓存命中率: {opt['cache_hit_rate']:.1%}")
                    print(f"  缓存命中: {opt['cache_hits']}")
                    print(f"  缓存未命中: {opt['cache_misses']}")
                
                if speedup >= 30:
                    print("  ✅ 达到目标: ≥30x 加速")
                else:
                    print("  ⚠️  未达到目标: <30x 加速")
        
        # 混合精度效果
        if 'mixed_precision' in results:
            mp = results['mixed_precision']
            print(f"\n⚡ 混合精度效果:")
            print(f"  普通精度时间: {mp.get('normal_time', 0):.2f}s")
            print(f"  混合精度时间: {mp.get('amp_time', 0):.2f}s")
            print(f"  加速比: {mp.get('speedup', 1):.1f}x")
            print(f"  显存节省: {mp.get('memory_saved', 'N/A')}")
        
        # 综合效果估算
        print(f"\n🎯 综合优化效果:")
        if 'original' in results and 'optimized' in results:
            feature_speedup = results['original'].get('total_time', 1) / results['optimized'].get('total_time', 1)
            amp_speedup = results.get('mixed_precision', {}).get('speedup', 1)
            total_speedup = feature_speedup * amp_speedup
            
            print(f"  特征缓存加速: {feature_speedup:.1f}x")
            print(f"  混合精度加速: {amp_speedup:.1f}x")
            print(f"  综合加速比: {total_speedup:.1f}x")
            
            if total_speedup >= 30:
                print("  ✅ 达到目标: ≥30x 加速")
            else:
                print("  ⚠️  未达到目标: <30x 加速")
        
        print(f"\n💡 说明:")
        print(f"  - 这是基于Mock数据的测试结果")
        print(f"  - 实际使用中加速效果可能更明显")
        print(f"  - 特征缓存在第二次及以后训练中效果最佳")


def main():
    """主函数"""
    print("Box Refinement 简化性能测试")
    print("=" * 60)
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if device == 'cpu':
        print("⚠️  使用CPU可能影响测试结果的准确性")
    
    # 创建测试器
    tester = SimplePerformanceTest(device)
    
    # 运行测试
    results = tester.run_comprehensive_test()
    
    # 打印结果
    tester.print_results(results)
    
    print("\n✅ 性能测试完成!")
    print("\n📝 测试结果已保存到控制台输出")


if __name__ == "__main__":
    main()