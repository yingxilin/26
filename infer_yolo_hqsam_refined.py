#!/usr/bin/env python3
"""
YOLOv8 + HQ-SAM + Box Refinement 推理脚本
在原有pipeline基础上增加Box Refinement模块

Usage:
    python infer_yolo_hqsam_refined.py \
        --yolo_weights "runs/detect/fungi_detection/weights/best.pt" \
        --ckpt_path "D:/search/fungi/26/data/models/fungitastic_ckpts" \
        --refinement_weights "checkpoints/box_refinement/best_model.pth" \
        --images_root "D:/search/fungi/26/data/FungiTastic-Mini/val/300p" \
        --out_masks "D:/search/fungi/26/FungiTastic/out/masks_yolo_hqsam_refined" \
        --sam_type "hq_vit_h" --conf 0.35 --iou 0.6
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# 确保项目根目录在路径中
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

# 导入我们的模块
from modules.box_refinement import BoxRefinementModule, visualize_refinement
from modules.hqsam_feature_extractor import create_hqsam_extractor

# 导入原有的HQ-SAM构建函数
sys.path.append(os.path.join(_project_root, 'segmentation'))
from hqsam.build_hqsam import build_sam_predictor


def xywh_norm_to_xyxy_abs(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """将归一化坐标转换为绝对坐标"""
    x_center = xc * img_w
    y_center = yc * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = int(round(x_center - bw / 2))
    y1 = int(round(y_center - bh / 2))
    x2 = int(round(x_center + bw / 2))
    y2 = int(round(y_center + bh / 2))
    return max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)


def run_yolo_batch(model, image_paths: List[Path], conf: float = 0.25, iou: float = 0.45, batch_size: int = 32):
    """批量运行YOLO检测"""
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batch_results = model.predict(batch, conf=conf, iou=iou, verbose=False)
        results.extend(batch_results)
    return results


def masks_postprocess(binary_mask: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    """后处理mask，去除小连通区域"""
    h, w = binary_mask.shape[:2]
    min_area = int(min_area_ratio * h * w)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary_mask > 0).astype(np.uint8), connectivity=8)
    output = np.zeros((h, w), dtype=np.uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == label] = 255
    return output


def save_mask(mask: np.ndarray, out_path: Path):
    """保存mask图像"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out_path)


def refine_bboxes_with_model(refinement_model, hqsam_extractor, image, bboxes, device):
    """
    使用Box Refinement模型精炼bboxes
    
    Args:
        refinement_model: 训练好的BoxRefinementModule
        hqsam_extractor: HQ-SAM特征提取器
        image: (H, W, 3) RGB图像
        bboxes: List[List[4]] - bbox列表
        device: 设备
    
    Returns:
        refined_bboxes: List[List[4]] - 精炼后的bbox列表
        refinement_history: List[List[List[4]]] - 每个bbox的迭代历史
    """
    if not bboxes:
        return [], []
    
    # 转换为tensor
    bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32, device=device)
    
    # 提取图像特征
    image_features = hqsam_extractor.extract_features(image)
    
    # 迭代精炼
    with torch.no_grad():
        refined_bboxes_tensor, history = refinement_model.iterative_refine(
            image_features, bboxes_tensor, image.shape[:2],
            max_iter=3, stop_threshold=1.0
        )
    
    # 转换回numpy
    refined_bboxes = refined_bboxes_tensor.cpu().numpy().tolist()
    refinement_history = [h.cpu().numpy().tolist() for h in history]
    
    return refined_bboxes, refinement_history


def main():
    print("Starting YOLOv8 + HQ-SAM + Box Refinement inference...", flush=True)
    
    parser = argparse.ArgumentParser(description="YOLOv8 + HQ-SAM + Box Refinement inference")
    
    # 原有参数
    parser.add_argument("--yolo_weights", required=True, help="Path to YOLOv8 weights .pt")
    parser.add_argument("--ckpt_path", required=True, help="SAM/HQ-SAM checkpoint path or directory")
    parser.add_argument("--images_root", required=True, help="Directory of images for inference")
    parser.add_argument("--out_masks", required=True, help="Output directory for masks")
    parser.add_argument("--sam_type", default="hq_vit_h", choices=["hq_vit_h", "hq_vit_l", "vit_h", "vit_l"], help="SAM model type")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="YOLO NMS IoU threshold")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--min_area_ratio", type=float, default=0.001, help="Minimum area ratio for CC filtering")
    parser.add_argument("--save_individual", action="store_true", help="Save one mask per image")
    
    # Box Refinement 参数
    parser.add_argument("--refinement_weights", help="Path to Box Refinement model weights")
    parser.add_argument("--enable_refinement", action="store_true", help="Enable Box Refinement")
    parser.add_argument("--refinement_config", default="configs/box_refinement_config.yaml", help="Box Refinement config file")
    parser.add_argument("--save_refinement_vis", action="store_true", help="Save refinement visualization")
    parser.add_argument("--refinement_vis_dir", help="Directory to save refinement visualizations")
    
    args = parser.parse_args()
    print(f"Arguments parsed successfully", flush=True)

    # 检查输入路径
    images_dir = Path(args.images_root)
    if not images_dir.exists():
        print(f"Error: images_root not found: {images_dir}")
        sys.exit(1)

    image_paths = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.JPG"), *images_dir.glob("*.png")])
    if not image_paths:
        print(f"No images found under {images_dir}")
        sys.exit(1)
    print(f"Found {len(image_paths)} images", flush=True)

    # 加载YOLO模型
    print(f"Loading YOLO model from {args.yolo_weights}...", flush=True)
    yolo_model = YOLO(args.yolo_weights)
    if args.device:
        yolo_model.to(args.device)
    print(f"Running YOLO detection on {len(image_paths)} images...", flush=True)
    yolo_results = run_yolo_batch(yolo_model, image_paths, conf=args.conf, iou=args.iou, batch_size=32)

    # 加载HQ-SAM predictor
    print(f"Loading HQ-SAM predictor ({args.sam_type})...", flush=True)
    predictor = build_sam_predictor(args.ckpt_path, sam_type=args.sam_type, device=args.device)
    print("HQ-SAM predictor loaded successfully", flush=True)

    # 加载Box Refinement模型（如果启用）
    refinement_model = None
    hqsam_extractor = None
    if args.enable_refinement and args.refinement_weights:
        print("Loading Box Refinement model...", flush=True)
        
        # 加载配置
        import yaml
        with open(args.refinement_config, 'r') as f:
            refinement_config = yaml.safe_load(f)
        
        # 创建模型
        refinement_model = BoxRefinementModule(
            hidden_dim=refinement_config['model']['hidden_dim'],
            num_heads=refinement_config['model']['num_heads'],
            max_offset=refinement_config['model']['max_offset']
        ).to(args.device)
        
        # 加载权重
        checkpoint = torch.load(args.refinement_weights, map_location=args.device)
        refinement_model.load_state_dict(checkpoint['model_state_dict'])
        refinement_model.eval()
        
        # 创建HQ-SAM特征提取器
        hqsam_extractor = create_hqsam_extractor(
            checkpoint_path=args.ckpt_path,
            model_type=args.sam_type,
            device=args.device,
            use_mock=False  # 使用真实的HQ-SAM
        )
        
        print("Box Refinement model loaded successfully", flush=True)

    # 创建输出目录
    out_root = Path(args.out_masks)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # 创建可视化目录（如果需要）
    if args.save_refinement_vis:
        vis_dir = Path(args.refinement_vis_dir) if args.refinement_vis_dir else out_root / "refinement_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

    # 处理每张图像
    for img_idx, (img_path, det) in enumerate(tqdm(zip(image_paths, yolo_results), desc="Processing images")):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"Warning: failed to read {img_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # 设置图像给SAM
        predictor.set_image(image_rgb)

        # 收集YOLO检测的boxes
        boxes_xyxy = []
        if det is not None and hasattr(det, "boxes") and det.boxes is not None:
            try:
                xyxy = det.boxes.xyxy.detach().cpu().numpy()
                for x1, y1, x2, y2 in xyxy:
                    x1c = int(max(0, min(w - 1, x1)))
                    y1c = int(max(0, min(h - 1, y1)))
                    x2c = int(max(0, min(w - 1, x2)))
                    y2c = int(max(0, min(h - 1, y2)))
                    if x2c > x1c and y2c > y1c:
                        boxes_xyxy.append([x1c, y1c, x2c, y2c])
            except Exception:
                pass

        if not boxes_xyxy:
            # 没有检测到目标，保存空mask
            empty = np.zeros((h, w), dtype=np.uint8)
            save_mask(empty, out_root / f"{img_path.stem}.png")
            continue

        # Box Refinement（如果启用）
        if refinement_model is not None:
            print(f"Refining {len(boxes_xyxy)} boxes for {img_path.name}...")
            refined_boxes, refinement_history = refine_bboxes_with_model(
                refinement_model, hqsam_extractor, image_rgb, boxes_xyxy, args.device
            )
            
            # 保存可视化（如果需要）
            if args.save_refinement_vis and refinement_history:
                # 选择第一个bbox进行可视化
                if refinement_history:
                    bbox_history = [torch.tensor([h[0]]) for h in refinement_history]
                    visualize_refinement(
                        image_rgb, bbox_history, 
                        vis_dir / f"{img_path.stem}_refinement.png"
                    )
            
            # 使用精炼后的boxes
            boxes_xyxy = refined_boxes

        # 使用SAM进行分割
        boxes_t = torch.tensor(boxes_xyxy, device=args.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_t, (h, w))

        with torch.no_grad():
            masks, scores, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

        # 合并实例masks
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in masks:
            m_np = m.squeeze(0).detach().cpu().numpy().astype(np.uint8) * 255
            combined = np.maximum(combined, m_np)

        # 后处理
        combined = masks_postprocess(combined, min_area_ratio=args.min_area_ratio)
        
        # 保存mask
        save_path = out_root / f"{img_path.stem}.png"
        save_mask(combined, save_path)
        
        # 保存单独的mask（如果需要）
        if args.save_individual:
            for i, m in enumerate(masks):
                m_np = m.squeeze(0).detach().cpu().numpy().astype(np.uint8) * 255
                m_np = masks_postprocess(m_np, min_area_ratio=args.min_area_ratio)
                individual_path = out_root / f"{img_path.stem}_instance_{i}.png"
                save_mask(m_np, individual_path)
        
    print(f"Inference completed. Masks saved to: {out_root}")
    if args.save_refinement_vis:
        print(f"Refinement visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()