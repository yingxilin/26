# Box Refinement Module 使用指南

## 概述

这个Box Refinement模块是对您现有YOLOv8 + HQ-SAM pipeline的改进，通过在HQ-SAM分割之前精炼YOLO检测的bbox来提升分割质量。

## 文件结构

```
/workspace/
├── modules/
│   ├── box_refinement.py          # 核心Box Refinement模块
│   └── hqsam_feature_extractor.py # HQ-SAM特征提取器
├── configs/
│   ├── box_refinement_config.yaml        # 完整配置
│   └── box_refinement_config_simple.yaml # 简化配置
├── train_box_refiner.py          # 训练脚本
├── test_box_refinement.py        # 测试脚本
├── infer_yolo_hqsam_refined.py   # 集成推理脚本
└── README_BoxRefinement.md       # 本文件
```

## 使用步骤

### 步骤1: 训练Box Refinement模型

首先需要训练Box Refinement模型：

```bash
# 使用micromamba环境
& C:\Users\23215\micromamba\Library\bin\micromamba.exe run -n fungitastic-seg python train_box_refiner.py --config configs/box_refinement_config_simple.yaml
```

**注意**: 在运行训练之前，请修改 `configs/box_refinement_config_simple.yaml` 中的路径：

```yaml
data:
  data_root: 'D:/search/fungi/26/data/FungiTastic-Mini'  # 修改为您的实际路径

hqsam:
  checkpoint: 'D:/search/fungi/26/data/models/fungitastic_ckpts'  # 修改为您的实际路径
```

### 步骤2: 测试训练好的模型

```bash
# 测试模型性能
& C:\Users\23215\micromamba\Library\bin\micromamba.exe run -n fungitastic-seg python test_box_refinement.py --config configs/box_refinement_config_simple.yaml --checkpoint checkpoints/box_refinement/best_model.pth
```

### 步骤3: 使用集成推理脚本

训练完成后，您可以使用集成的推理脚本，它结合了YOLO检测、Box Refinement和HQ-SAM分割：

```bash
# 不使用Box Refinement（基线）
& C:\Users\23215\micromamba\Library\bin\micromamba.exe run -n fungitastic-seg python infer_yolo_hqsam_refined.py \
    --yolo_weights "runs\detect\fungi_detection\weights\best.pt" \
    --ckpt_path "D:\search\fungi\26\data\models\fungitastic_ckpts" \
    --images_root "D:\search\fungi\26\data\FungiTastic-Mini\val\300p" \
    --out_masks "D:\search\fungi\26\FungiTastic\out\masks_yolo_hqsam_baseline" \
    --sam_type "hq_vit_h" --conf 0.35 --iou 0.6 --min_area_ratio 0.001 --device "cuda"

# 使用Box Refinement（改进版）
& C:\Users\23215\micromamba\Library\bin\micromamba.exe run -n fungitastic-seg python infer_yolo_hqsam_refined.py \
    --yolo_weights "runs\detect\fungi_detection\weights\best.pt" \
    --ckpt_path "D:\search\fungi\26\data\models\fungitastic_ckpts" \
    --refinement_weights "checkpoints/box_refinement/best_model.pth" \
    --images_root "D:\search\fungi\26\data\FungiTastic-Mini\val\300p" \
    --out_masks "D:\search\fungi\26\FungiTastic\out\masks_yolo_hqsam_refined" \
    --sam_type "hq_vit_h" --conf 0.35 --iou 0.6 --min_area_ratio 0.001 --device "cuda" \
    --enable_refinement --save_refinement_vis --refinement_vis_dir "D:\search\fungi\26\FungiTastic\out\refinement_visualizations"
```

## 参数说明

### 新增参数

- `--refinement_weights`: Box Refinement模型权重路径
- `--enable_refinement`: 启用Box Refinement功能
- `--refinement_config`: Box Refinement配置文件路径
- `--save_refinement_vis`: 保存精炼过程可视化
- `--refinement_vis_dir`: 可视化结果保存目录

### 原有参数保持不变

所有原有的YOLO和HQ-SAM参数都保持不变，确保兼容性。

## 预期效果

1. **训练阶段**: Box Refinement模型学习如何从YOLO的粗糙bbox预测更精确的bbox
2. **推理阶段**: 
   - 基线: YOLO bbox → HQ-SAM分割
   - 改进: YOLO bbox → Box Refinement → 精炼bbox → HQ-SAM分割

3. **预期提升**: IoU从90.22%提升到92-95%

## 调试模式

如果遇到问题，可以使用调试模式：

```bash
# 训练调试模式
& C:\Users\23215\micromamba\Library\bin\micromamba.exe run -n fungitastic-seg python train_box_refiner.py --config configs/box_refinement_config_simple.yaml --debug

# 测试调试模式
& C:\Users\23215\micromamba\Library\bin\micromamba.exe run -n fungitastic-seg python test_box_refinement.py --config configs/box_refinement_config_simple.yaml --checkpoint checkpoints/box_refinement/best_model.pth --debug
```

## 输出文件

### 训练输出
- `checkpoints/box_refinement/best_model.pth`: 最佳模型权重
- `logs/box_refinement/training.log`: 训练日志
- `visualizations/box_refinement/`: 训练过程可视化

### 测试输出
- `outputs/box_refinement/evaluation_results.txt`: 评估结果
- `outputs/box_refinement/evaluation_metrics.png`: 指标图表
- `visualizations/box_refinement/sample_*.png`: 样本可视化

### 推理输出
- 分割masks (与原pipeline相同)
- `refinement_visualizations/`: Box精炼过程可视化

## 故障排除

1. **路径问题**: 确保所有路径都正确设置
2. **内存不足**: 减小batch_size或使用更小的模型
3. **CUDA问题**: 确保CUDA环境正确配置
4. **依赖问题**: 确保所有必要的包都已安装

## 性能优化

1. **训练速度**: 使用Mock HQ-SAM进行快速测试
2. **推理速度**: Box Refinement只增加很少的计算开销
3. **内存使用**: 可以调整batch_size和图像尺寸