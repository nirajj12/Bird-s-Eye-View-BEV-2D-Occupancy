# data/preprocess.py
# ══════════════════════════════════════════════════════
# Preprocessing functions for BEV Occupancy Project
# Handles: images, camera params, LiDAR → BEV grid
# ══════════════════════════════════════════════════════

import numpy as np
import cv2
import torch
from pyquaternion import Quaternion

from config.config import (
    IMG_H, IMG_W,
    IMG_MEAN, IMG_STD,
    BEV_H, BEV_W,
    X_RANGE, Y_RANGE
)
from logger.custom_logger import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


# ──────────────────────────────────────────────────────
# 1. Image preprocessing
# ──────────────────────────────────────────────────────

def preprocess_image(img_path: str) -> torch.Tensor:
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(
                f"Image not found at: {img_path}"
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img,
            (IMG_W, IMG_H),
            interpolation=cv2.INTER_LINEAR
        )

        img  = img.astype(np.float32) / 255.0
        mean = np.array(IMG_MEAN, dtype=np.float32)
        std  = np.array(IMG_STD,  dtype=np.float32)
        img  = (img - mean) / std
        img  = img.transpose(2, 0, 1)   # HWC → CHW

        tensor = torch.tensor(img, dtype=torch.float32)

        # ✅ FIX: debug not info — avoids 116k log lines per training run
        logger.debug(f"Image preprocessed | shape: {tuple(tensor.shape)}")

        return tensor

    except Exception as e:
        raise BEVException(
            f"Failed to preprocess image: {img_path}", e
        ) from e


# ──────────────────────────────────────────────────────
# 2. Camera parameter preprocessing
# ──────────────────────────────────────────────────────

def preprocess_intrinsic(
    camera_intrinsic: list,
    orig_w: int,
    orig_h: int
) -> torch.Tensor:
    try:
        K = np.array(camera_intrinsic, dtype=np.float32)

        scale_w = IMG_W / orig_w
        scale_h = IMG_H / orig_h

        K[0, 0] *= scale_w   # fx
        K[0, 2] *= scale_w   # cx
        K[1, 1] *= scale_h   # fy
        K[1, 2] *= scale_h   # cy

        # ✅ FIX: debug not info
        logger.debug("Intrinsic K scaled for resized image")

        return torch.tensor(K, dtype=torch.float32)

    except Exception as e:
        raise BEVException(
            "Failed to preprocess camera intrinsic", e
        ) from e


def preprocess_extrinsic(
    rotation: list,
    translation: list
) -> torch.Tensor:
    try:
        rot   = Quaternion(rotation).rotation_matrix
        trans = np.array(translation, dtype=np.float32)

        E = np.eye(4, dtype=np.float32)
        E[:3, :3] = rot
        E[:3,  3] = trans

        # ✅ FIX: debug not info
        logger.debug("Extrinsic E matrix built")

        return torch.tensor(E, dtype=torch.float32)

    except Exception as e:
        raise BEVException(
            "Failed to preprocess camera extrinsic", e
        ) from e


# ──────────────────────────────────────────────────────
# 3. LiDAR → BEV occupancy grid
# ──────────────────────────────────────────────────────

def build_bev_occupancy(
    lidar_points: np.ndarray,
    lidar_rotation: list,
    lidar_translation: list
) -> torch.Tensor:
    try:
        points = lidar_points[:3, :].T    # (N, 3)

        rot    = Quaternion(lidar_rotation).rotation_matrix
        trans  = np.array(lidar_translation, dtype=np.float32)
        points = (rot @ points.T).T + trans

        # Z-height filter
        z_ego       = points[:, 2]
        height_mask = (z_ego > 0.2) & (z_ego < 3.5)
        points      = points[height_mask]

        occ   = np.zeros((BEV_H, BEV_W), dtype=np.float32)
        x_res = (X_RANGE[1] - X_RANGE[0]) / BEV_W
        y_res = (Y_RANGE[1] - Y_RANGE[0]) / BEV_H

        col = ((points[:, 0] - X_RANGE[0]) / x_res).astype(int)
        row = ((points[:, 1] - Y_RANGE[0]) / y_res).astype(int)

        mask = (col >= 0) & (col < BEV_W) & (row >= 0) & (row < BEV_H)
        occ[row[mask], col[mask]] = 1.0

        occupied = int(occ.sum())
        total    = BEV_H * BEV_W

        # ✅ FIX: debug not info
        logger.debug(
            f"BEV GT built | occupied: {occupied}/{total} "
            f"({100 * occupied / total:.1f}%) | after Z-filter"
        )

        return torch.tensor(occ, dtype=torch.float32)

    except Exception as e:
        raise BEVException(
            "Failed to build BEV occupancy grid", e
        ) from e