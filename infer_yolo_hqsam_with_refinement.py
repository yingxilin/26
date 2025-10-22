#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HQ-SAM + YOLOv8 推理脚本（修正版 v12 - 修复 RLE 解码）
"""

import os, sys, cv2, torch, numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import gc

SAM_HQ_PATH = r"d:\search\fungi\26\sam-hq"
if SAM_HQ_PATH not in sys.path:
    sys.path.append(SAM_HQ_PATH)

from segment_anything import sam_model_registry

# ========================== 配置 ==========================
YOLO_WEIGHTS = r"D:\search\fungi\26\FungiTastic\runs\detect\fungi_detection\weights\best.pt"
SAM_CKPT     = r"D:\search\fungi\26\data\models\fungitastic_ckpts\sam_hq_vit_h.pth"
IMAGES_ROOT  = r"D:\search\fungi\26\data\FungiTastic-Mini\val\300p"
GT_PARQUET   = r"D:\search\fungi\26\data\masks\FungiTastic-Mini-ValidationMasks.parquet"
OUT_ROOT     = r"D:\search\fungi\26_2\26\gaijinout\masksyolo_hqsam_final"
DEVICE = "cuda"
YOLO_CONF = 0.35
BATCH_SIZE = 50
# =========================================================

os.makedirs(OUT_ROOT, exist_ok=True)
LOG_PATH = os.path.join(OUT_ROOT, "inference_log.txt")


def rle_decode_fixed(rle_array, height, width):
    """
    修复的 RLE 解码
    RLE 可能的格式：
    1. [start1, length1, start2, length2, ...] - 从 0 开始的索引
    2. [start1, length1, start2, length2, ...] - 从 1 开始的索引（COCO 格式）
    """
    try:
        if not isinstance(rle_array, np.ndarray) or len(rle_array) == 0:
            return None
        
        # 确保是偶数长度
        if len(rle_array) % 2 != 0:
            return None
        
        total_size = height * width
        mask = np.zeros(total_size, dtype=np.uint8)
        
        # 检查是否是 1-indexed（COCO 格式）
        # 如果第一个 start 是 0，可能是 0-indexed
        # 如果第一个 start >= 1，可能是 1-indexed
        first_start = int(rle_array[0])
        is_one_indexed = first_start >= 1
        
        decoded_pixels = 0
        
        for i in range(0, len(rle_array), 2):
            start = int(rle_array[i])
            length = int(rle_array[i + 1])
            
            # 如果是 1-indexed，转换为 0-indexed
            if is_one_indexed:
                start = start - 1
            
            # 边界检查
            if start < 0 or length <= 0:
                continue
            
            if start >= total_size:
                continue
            
            end = min(start + length, total_size)
            mask[start:end] = 255
            decoded_pixels += (end - start)
        
        # 如果解码的像素太少，可能是格式问题
        if decoded_pixels < 10:
            return None
        
        return mask.reshape(height, width)
        
    except Exception as e:
        return None


def normalize_filename(fname):
    """标准化文件名"""
    p = Path(fname)
    stem = p.stem.lower()
    return stem


def load_gt_masks_from_parquet(parquet_path):
    """从 parquet 加载 GT 掩码"""
    print(f"📖 Loading GT masks from: {parquet_path}")
    
    try:
        df = pd.read_parquet(parquet_path)
        print(f"   Total rows: {len(df)}")
        
        # 调试：查看第一个 RLE 的内容
        print(f"\n   🔍 Debugging first RLE:")
        first_rle = df.iloc[0]['rle']
        print(f"     Type: {type(first_rle)}")
        print(f"     Shape: {first_rle.shape if isinstance(first_rle, np.ndarray) else 'N/A'}")
        print(f"     First 10 values: {first_rle[:10] if isinstance(first_rle, np.ndarray) else 'N/A'}")
        print(f"     Min/Max: {first_rle.min()}/{first_rle.max() if isinstance(first_rle, np.ndarray) else 'N/A'}")
        
        # 按文件名分组
        gt_dict = {}
        grouped = df.groupby('file_name')
        
        print(f"\n   Processing {len(grouped)} unique images...")
        
        success_count = 0
        fail_count = 0
        fail_details = {'no_decode': 0, 'empty_mask': 0, 'exception': 0}
        
        # 测试前几个
        test_results = []
        
        for idx, (filename, group) in enumerate(tqdm(grouped, desc="   Decoding GT", leave=False)):
            try:
                filename_key = normalize_filename(str(filename))
                
                first_row = group.iloc[0]
                H = int(first_row['height'])
                W = int(first_row['width'])
                
                combined_mask = np.zeros((H, W), dtype=np.uint8)
                part_success = 0
                part_fail = 0
                
                for _, row in group.iterrows():
                    rle_data = row['rle']
                    
                    if isinstance(rle_data, np.ndarray) and len(rle_data) > 0:
                        part_mask = rle_decode_fixed(rle_data, H, W)
                        
                        if part_mask is not None:
                            combined_mask = np.maximum(combined_mask, part_mask)
                            part_success += 1
                        else:
                            part_fail += 1
                
                # 记录测试结果
                if idx < 5:
                    test_results.append({
                        'filename': filename,
                        'shape': (H, W),
                        'parts': len(group),
                        'success': part_success,
                        'fail': part_fail,
                        'mask_pixels': int(combined_mask.sum() / 255)
                    })
                
                if combined_mask.max() > 0:
                    gt_dict[filename_key] = combined_mask
                    success_count += 1
                else:
                    fail_count += 1
                    if part_success == 0:
                        fail_details['no_decode'] += 1
                    else:
                        fail_details['empty_mask'] += 1
                    
            except Exception as e:
                fail_count += 1
                fail_details['exception'] += 1
                continue
        
        print(f"\n   ✅ Loaded: {success_count}, Failed: {fail_count}")
        print(f"   Failure breakdown:")
        print(f"     - No parts decoded: {fail_details['no_decode']}")
        print(f"     - Empty after decode: {fail_details['empty_mask']}")
        print(f"     - Exceptions: {fail_details['exception']}")
        
        print(f"\n   Test results (first 5):")
        for r in test_results:
            print(f"     {r['filename']}: {r['shape']}, {r['parts']} parts, "
                  f"{r['success']} success, {r['fail']} fail, {r['mask_pixels']} pixels")
        
        if len(gt_dict) > 0:
            sample_keys = list(gt_dict.keys())[:5]
            print(f"\n   Sample GT keys: {sample_keys}")
        
        return gt_dict
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def compute_iou(pred_mask, gt_mask):
    """计算 IoU"""
    p = (pred_mask > 127).astype(np.uint8)
    g = (gt_mask > 127).astype(np.uint8)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    print("\n🔹 Loading YOLO...")
    yolo = YOLO(YOLO_WEIGHTS)
    yolo.to(device).eval()

    print("🔹 Loading HQ-SAM...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT).to(device).eval()

    print()
    gt_masks = {}
    if Path(GT_PARQUET).exists():
        gt_masks = load_gt_masks_from_parquet(GT_PARQUET)
    else:
        print(f"⚠️ GT not found")

    print("\n🔍 Scanning images...")
    
    image_dict = {}
    for pattern in ["*.jpg", "*.JPG", "*.png", "*.PNG"]:
        for p in Path(IMAGES_ROOT).glob(pattern):
            key = normalize_filename(p.name)
            if key not in image_dict:
                image_dict[key] = p
    
    image_paths = sorted(image_dict.values(), key=lambda x: x.name)
    print(f"   Found {len(image_paths)} unique images")
    
    if len(gt_masks) > 0:
        img_keys = set(image_dict.keys())
        gt_keys = set(gt_masks.keys())
        matched = img_keys.intersection(gt_keys)
        
        print(f"\n   Filename matching:")
        print(f"     Images: {len(img_keys)}")
        print(f"     GT masks: {len(gt_keys)}")
        print(f"     ✅ Matched: {len(matched)}")

    print(f"\n{'='*60}")
    print(f"🔬 Starting inference...")
    print(f"{'='*60}\n")
    
    ious, strengths = [], []
    no_detection = 0
    gt_found = 0
    failed = 0

    pbar = tqdm(image_paths, desc="Processing")
    
    for idx, img_path in enumerate(pbar):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                failed += 1
                continue
            
            H, W = img.shape[:2]

            with torch.no_grad():
                res = yolo.predict(source=str(img_path), conf=YOLO_CONF, verbose=False)[0]
                boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else np.array([])

            if len(boxes) == 0:
                no_detection += 1
                cv2.imwrite(str(Path(OUT_ROOT) / img_path.name), np.zeros((H, W), np.uint8))
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.as_tensor(img_rgb, device=device).permute(2, 0, 1).float() / 255.0
            img_1024 = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0), size=(1024, 1024), mode="bilinear", align_corners=False
            )

            with torch.no_grad():
                encoder_out = sam.image_encoder(img_1024)
                if isinstance(encoder_out, (tuple, list)):
                    image_emb = encoder_out[0]
                    interm_emb = encoder_out[1] if len(encoder_out) > 1 else None
                else:
                    image_emb = encoder_out
                    interm_emb = None

            boxes_1024 = boxes.copy()
            boxes_1024[:, [0, 2]] *= (1024.0 / W)
            boxes_1024[:, [1, 3]] *= (1024.0 / H)

            with torch.no_grad():
                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=None,
                    boxes=torch.tensor(boxes_1024, device=device, dtype=torch.float32),
                    masks=None,
                )
                
                mask_logits, _ = sam.mask_decoder(
                    image_embeddings=image_emb,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                    hq_token_only=False,
                    interm_embeddings=interm_emb,
                )

            masks = torch.sigmoid(mask_logits).cpu().numpy()
            
            combined = np.zeros((256, 256), dtype=np.float32)
            for m in masks:
                combined = np.maximum(combined, m[0])
            
            combined = cv2.resize(combined, (W, H), interpolation=cv2.INTER_LINEAR)
            final = (combined * 255).astype(np.uint8)
            
            if final.max() > 0:
                final = cv2.normalize(final, None, 0, 255, cv2.NORM_MINMAX)
            
            _, final = cv2.threshold(final, 127, 255, cv2.THRESH_BINARY)
            
            cv2.imwrite(str(Path(OUT_ROOT) / img_path.name), final)
            strengths.append(float(final.mean()))

            key = normalize_filename(img_path.name)
            if key in gt_masks:
                gt = gt_masks[key]
                if gt.shape[:2] != (H, W):
                    gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
                
                iou = compute_iou(final, gt)
                ious.append(iou)
                gt_found += 1
            
            if (idx + 1) % BATCH_SIZE == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            pbar.set_postfix({
                'no_det': no_detection,
                'gt_match': gt_found,
                'iou': f'{np.mean(ious):.3f}' if ious else 'N/A'
            })

        except Exception as e:
            failed += 1
            try:
                cv2.imwrite(str(Path(OUT_ROOT) / img_path.name), np.zeros((H, W), np.uint8))
            except:
                pass
            continue

    mean_iou = np.mean(ious) if ious else 0.0
    median_iou = np.median(ious) if ious else 0.0
    mean_str = np.mean(strengths) if strengths else 0.0
    
    summary = f"""
{'='*60}
🎯 Results
{'='*60}
Total:              {len(image_paths)}
Processed:          {len(image_paths) - failed}
No detection:       {no_detection}
Failed:             {failed}
GT matched:         {gt_found}

Mean IoU:           {mean_iou:.4f}
Median IoU:         {median_iou:.4f}
Mask intensity:     {mean_str:.2f}

Output: {OUT_ROOT}
{'='*60}
"""
    
    print(f"\n{summary}")
    
    with open(LOG_PATH, "w") as f:
        f.write(summary)
        if ious:
            f.write(f"\nIoU Stats:\n")
            f.write(f"Min:  {np.min(ious):.4f}\n")
            f.write(f"Max:  {np.max(ious):.4f}\n")
            f.write(f"Std:  {np.std(ious):.4f}\n")
    
    print(f"📝 Log: {LOG_PATH}")


if __name__ == "__main__":
    main()