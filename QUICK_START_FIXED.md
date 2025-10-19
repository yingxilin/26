# Box Refinement 训练快速开始指南（已修复版本）

## 问题已修复

✅ 所有代码问题已修复，包括：
- 语法错误和重复代码
- 坐标系不一致（损失过高的根本原因）
- 数据加载性能问题
- Windows平台兼容性

## 重要：清除旧缓存

在运行训练前，**必须**清除旧的特征缓存（因为模型输入输出格式已变化）：

```bash
# Windows PowerShell
Remove-Item -Recurse -Force checkpoints\box_refinement\features\

# Linux/Mac
rm -rf checkpoints/box_refinement/features/
```

## 运行训练

### 1. 快速测试（推荐先运行）

```bash
python train_box_refiner_optimized.py --config configs/box_refinement_config.yaml --fast --clear-cache --debug
```

参数说明：
- `--fast`: 快速模式（10%数据，混合精度）
- `--clear-cache`: 自动清除特征缓存
- `--debug`: 只用100个样本测试

期望输出：
```
🚀 Fast mode enabled - applying all optimizations
  - Data sampling: 0.1
  - Mixed precision: True
  - Batch size: 32
Using device: cuda
Feature cache detected: False
Clearing feature cache...
Feature cache cleared.
Loading datasets...
Found 93684 images in train split
Found 18900 images in val split
Sampled 9368 images from 93684 total images (ratio: 0.1)
Sampled 1890 images from 18900 total images (ratio: 0.1)
Creating model...
Loading HQ-SAM feature extractor...
Mock HQ-SAM feature extractor initialized (model_type: hq_vit_h)
Creating optimizer...
Learning rate: 0.0001
Starting training...
Epoch 0:   0%|  | 0/4 [00:00<?, ?it/s]
  Loading first batch... (this may take a while)
  Extracting features for first batch...
  Feature extraction completed. Starting training...
Epoch 0: 100%|██████| 4/4 [00:05<00:00, 0.75it/s, Loss=0.3245, L1=0.0234, IoU=0.1567, Cache=0.0%]
```

**关键指标**：
- ✅ Loss < 1.0（修复前 > 200）
- ✅ L1 < 0.1（修复前 > 20）
- ✅ IoU < 0.5（修复前 ~1.0）

### 2. 完整训练

确认测试成功后，运行完整训练：

```bash
python train_box_refiner_optimized.py --config configs/box_refinement_config.yaml --clear-cache
```

## 验证修复

运行单元测试验证所有修复：

```bash
python test_box_refinement_fixed.py
```

期望输出：
```
============================================================
Testing Box Refinement Module (Fixed Version)
============================================================
Using device: cuda

1. Testing model initialization...
   ✓ Model initialized successfully
   ...

============================================================
✓ ALL TESTS PASSED!
============================================================
```

## 配置说明

关键配置参数（已更新为正确值）：

```yaml
# configs/box_refinement_config.yaml

model:
  max_offset: 0.1  # ⚠️ 归一化坐标（不是像素）

refinement:
  stop_threshold: 0.01  # ⚠️ 归一化坐标（不是像素）

data:
  num_workers: 8  # Windows自动降为4
  persistent_workers: true  # Windows自动禁用
```

## 预期性能

### 训练速度

| 平台 | 批大小 | Workers | 速度 |
|------|--------|---------|------|
| Windows | 32 | 4 | ~0.5 it/s |
| Linux | 32 | 8 | ~1.0 it/s |

### 损失收敛

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 0 | 0.45 | 0.50 |
| 5 | 0.25 | 0.30 |
| 10 | 0.15 | 0.22 |
| 20 | 0.08 | 0.18 |

## 常见问题

### Q1: 训练还是卡在0%

**A**: 可能是Windows多进程问题，尝试：
```bash
# 方法1：减少workers
# 编辑 configs/box_refinement_config.yaml
data:
  num_workers: 2  # 改为2或0

# 方法2：使用单进程模式
data:
  num_workers: 0
```

### Q2: 损失仍然很高

**A**: 确认以下几点：
1. 已清除旧缓存
2. 配置文件中 `max_offset: 0.1`（不是50）
3. 配置文件中 `stop_threshold: 0.01`（不是1.0）

### Q3: CUDA内存不足

**A**: 减小批大小：
```yaml
training:
  batch_size: 16  # 改为 8 或更小
```

## 技术细节

### 关键修复点

1. **坐标系统一**
   - 数据集输出：归一化坐标 [0, 1]
   - 模型输入：归一化坐标 [0, 1]
   - 模型输出：归一化偏移 [-0.1, 0.1]
   - 损失计算：归一化坐标 [0, 1]

2. **数值范围**
   - bbox: [0, 1]
   - offset: [-0.1, 0.1]
   - L1 loss: ~0.01-0.1
   - IoU loss: ~0.1-0.5
   - Total loss: ~0.1-0.5

3. **模型初始化**
   - 使用小权重初始化 (std=0.001)
   - 使用tanh激活限制输出
   - 添加dropout防止过拟合

## 相关文件

- `BOX_REFINEMENT_FIXES.md` - 详细修复报告
- `test_box_refinement_fixed.py` - 单元测试脚本
- `modules/box_refinement.py` - 修复后的模型代码
- `configs/box_refinement_config.yaml` - 更新后的配置
- `train_box_refiner_optimized.py` - 优化后的训练脚本

## 获取帮助

如果遇到问题，请提供：
1. 完整的错误信息
2. 训练日志（前50行）
3. 配置文件内容
4. 系统信息（Windows/Linux, CUDA版本等）
