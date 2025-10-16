#!/usr/bin/env python3
"""
Utility to build SAM or HQ-SAM predictors from local checkpoints for use in the
YOLOv8-det → HQ-SAM pipeline.

Exposes a single factory `build_sam_predictor` returning a SegmentAnything
predictor-like object with `.set_image(np.ndarray)` and `.predict_torch`/`.predict`
APIs used by common SAM integrations.

Supported `sam_type` values:
  - "hq_vit_h", "hq_vit_l": HQ-SAM models
  - "vit_h", "vit_l": standard Meta SAM models

The function searches for checkpoint files under the provided `ckpt_path`. For
standard SAM you can pass the full file path to the .pth file.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch


def _import_sam_modules(sam_type: str):
    """Import SAM or HQ-SAM modules lazily based on type."""
    # HQ-SAM uses the same segment_anything namespace when installed from sam-hq
    try:
        from segment_anything import sam_model_registry
        from segment_anything import SamPredictor
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "segment-anything not installed. Install HQ-SAM from sam-hq directory: "
            "pip install -e /path/to/sam-hq"
        ) from exc
    return sam_model_registry, SamPredictor


def _resolve_checkpoint(ckpt_path: str, sam_type: str) -> str:
    """Resolve checkpoint path. Accepts a directory or a specific .pth file."""
    p = Path(ckpt_path)
    if p.is_file():
        return str(p)

    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")

    # Try common filenames
    candidates = []
    if sam_type.startswith("hq_"):
        # HQ-SAM checkpoints are usually named like hq_sam_{vit_h|vit_l}.pth
        candidates += list(p.glob("**/hq_sam_*.pth"))
        candidates += list(p.glob("**/*hq*.pth"))
    else:
        candidates += list(p.glob("**/sam_vit_*.pth"))

    if not candidates:
        # Fallback to any .pth
        candidates = list(p.glob("**/*.pth"))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint .pth found under {ckpt_path} for {sam_type}"
        )

    # Prefer largest file (likely full checkpoint)
    best = max(candidates, key=lambda x: x.stat().st_size)
    return str(best)


def _model_name_from_type(sam_type: str) -> str:
    if sam_type in ("hq_vit_h", "vit_h"):
        return "vit_h"
    if sam_type in ("hq_vit_l", "vit_l"):
        return "vit_l"
    raise ValueError(f"Unsupported sam_type: {sam_type}")


def build_sam_predictor(
    ckpt_path: str,
    sam_type: str = "hq_vit_h",
    device: Optional[str] = None,
):
    """
    Build a SAM/HQ-SAM predictor.

    Args:
        ckpt_path: Path to directory containing checkpoints or full .pth file
        sam_type: One of {hq_vit_h, hq_vit_l, vit_h, vit_l}
        device: cuda/cpu. If None, auto-select CUDA when available.

    Returns:
        predictor: Initialized predictor ready for set_image/predict calls
    """
    registry, PredictorCls = _import_sam_modules(sam_type)
    model_name = _model_name_from_type(sam_type)
    ckpt = _resolve_checkpoint(ckpt_path, sam_type)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Both regular SAM and HQ-SAM use the same model names in the registry
    # When HQ-SAM is installed, it replaces regular SAM with HQ versions
    build_name = model_name
    sam = registry[build_name](checkpoint=ckpt)

    sam.to(device=device)
    predictor = PredictorCls(sam)
    return predictor


__all__ = ["build_sam_predictor"]


