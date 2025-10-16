#!/usr/bin/env python3
"""
YOLOv8 + HQ-SAM + Box Refinement 完整推理脚本
在原有pipeline基础上增加Box Refinement模块

Usage:
    python infer_yolo_hqsam_with_refinement.py \
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

# 导入Box Refinement模块
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


def main():
    print("Starting YOLOv8 + HQ-SAM + Box Refinement inference...", flush=True)
    
    parser = argparse.ArgumentParser(description="YOLOv8 + HQ-SAM + Box Refinement inference")
    
    # YOLO参数
    parser.add_argument("--yolo_weights", required=True, help="Path to YOLOv8 weights .pt")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="YOLO NMS IoU threshold")
    
    # HQ-SAM参数
    parser.add_argument("--ckpt_path", required=True, help="SAM/HQ-SAM checkpoint path or directory")
    parser.add_argument("--sam_type", default="hq_vit_h", choices=["hq_vit_h", "hq_vit_l", "vit_h", "vit_l"], help="SAM model type")
    
    # Box Refinement参数
    parser.add_argument("--refinement_weights", required=True, help="Path to Box Refinement model weights")
    parser.add_argument("--refinement_config", help="Path to Box Refinement config file (optional)")
    parser.add_argument("--max_iter", type=int, default=3, help="Maximum refinement iterations")
    parser.add_argument("--stop_threshold", type=float, default=1.0, help="Early stopping threshold (pixels)")
    
    # 输入输出参数
    parser.add_argument("--images_root", required=True, help="Directory of images for inference")
    parser.add_argument("--out_masks", required=True, help="Output directory for masks")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--min_area_ratio", type=float, default=0.001, help="Minimum area ratio for CC filtering")
    parser.add_argument("--save_individual", action="store_true", help="Save one mask per image")
    
    # 可视化参数
    parser.add_argument("--save_visualizations", action="store_true", help="Save refinement visualizations")
    parser.add_argument("--vis_dir", help="Directory to save visualizations")
    
    args = parser.parse_args()
    print(f"Arguments parsed successfully", flush=True)

    # 检查输入目录
    images_dir = Path(args.images_root)
    if not images_dir.exists():
        print(f"Error: images_root not found: {images_dir}")
        sys.exit(1)

    image_paths = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.JPG"), *images_dir.glob("*.png")])
    if not image_paths:
        print(f"No images found under {images_dir}")
        sys.exit(1)
    print(f"Found {len(image_paths)} images", flush=True)

    # 加载Box Refinement模型
    print(f"Loading Box Refinement model from {args.refinement_weights}...", flush=True)
    refinement_model = BoxRefinementModule(
        hidden_dim=256,
        num_heads=8,
        max_offset=50
    ).to(args.device)
    
    # 加载权重
    checkpoint = torch.load(args.refinement_weights, map_location=args.device)
    refinement_model.load_state_dict(checkpoint['model_state_dict'])
    refinement_model.eval()
    print("Box Refinement model loaded successfully", flush=True)

    # 加载HQ-SAM特征提取器
    print(f"Loading HQ-SAM feature extractor...", flush=True)
    hqsam_extractor = create_hqsam_extractor(
        checkpoint_path=args.ckpt_path,
        model_type=args.sam_type,
        device=args.device,
        use_mock=False  # 使用真实的HQ-SAM
    )
    print("HQ-SAM feature extractor loaded successfully", flush=True)

    # 加载HQ-SAM分割器
    print(f"Loading HQ-SAM predictor ({args.sam_type})...", flush=True)
    sam_predictor = build_sam_predictor(args.ckpt_path, sam_type=args.sam_type, device=args.device)
    print("HQ-SAM predictor loaded successfully", flush=True)

    # 加载YOLO模型
    print(f"Loading YOLO model from {args.yolo_weights}...", flush=True)
    yolo_model = YOLO(args.yolo_weights)
    if args.device:
        yolo_model.to(args.device)
    print(f"Running YOLO detection on {len(image_paths)} images...", flush=True)
    yolo_results = run_yolo_batch(yolo_model, image_paths, conf=args.conf, iou=args.iou, batch_size=32)

    # 创建输出目录
    out_root = Path(args.out_masks)
    out_root.mkdir(parents=True, exist_ok=True)
    
    if args.save_visualizations:
        vis_dir = Path(args.vis_dir) if args.vis_dir else out_root / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

    # 处理每张图像
    print("Processing images with Box Refinement...", flush=True)
    
    for img_idx, (img_path, det) in enumerate(tqdm(zip(image_paths, yolo_results), total=len(image_paths))):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"Warning: failed to read {img_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # 设置图像给SAM
        sam_predictor.set_image(image_rgb)

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

        # 转换为tensor
        boxes_t = torch.tensor(boxes_xyxy, device=args.device, dtype=torch.float32)
        
        # 使用Box Refinement精炼bbox
        print(f"Refining {len(boxes_xyxy)} boxes for image {img_idx+1}/{len(image_paths)}...", flush=True)
        
        with torch.no_grad():
            # 提取图像特征
            image_features = hqsam_extractor.extract_features(image_rgb)  # (1, 256, 64, 64)
            
            # 扩展特征以匹配bbox数量
            if len(boxes_xyxy) > 1:
                image_features = image_features.repeat(len(boxes_xyxy), 1, 1, 1)  # (B, 256, 64, 64)
            
            # 迭代精炼bbox
            refined_boxes, refinement_history = refinement_model.iterative_refine(
                image_features, boxes_t, (h, w),
                max_iter=args.max_iter,
                stop_threshold=args.stop_threshold
            )
            
            # 转换回CPU numpy数组
            refined_boxes_np = refined_boxes.cpu().numpy()
        
        # 使用精炼后的bbox进行HQ-SAM分割
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(refined_boxes, (h, w))

        with torch.no_grad():
            masks, scores, _ = sam_predictor.predict_torch(
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
        
        # 保存可视化（如果启用）
        if args.save_visualizations and len(boxes_xyxy) > 0:
            # 选择第一个bbox进行可视化
            bbox_history = [boxes_t[0:1].cpu().numpy()] + [h[0:1].cpu().numpy() for h in refinement_history[1:]]
            vis_path = vis_dir / f"{img_path.stem}_refinement.png"
            visualize_refinement(
                image_rgb, bbox_history, str(vis_path),
                gt_bbox=None  # 没有ground truth
            )

    print(f"Inference completed! Masks saved to: {out_root}")
    if args.save_visualizations:
        print(f"Visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()