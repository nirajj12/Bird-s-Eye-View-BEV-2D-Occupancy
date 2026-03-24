# data/nuscenes_loader.py
# ══════════════════════════════════════════════════════
# PyTorch Dataset + DataLoader for nuScenes BEV project
# V3: returns full_dataset + val_ds for per-sample eval
# ══════════════════════════════════════════════════════

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from config.config import (
    DATAROOT, VERSION,
    CAM_NAMES,
    BATCH_SIZE, NUM_WORKERS,
    TRAIN_SPLIT,
    NUM_SWEEPS
)
from data.preprocess import (
    preprocess_image,
    preprocess_intrinsic,
    preprocess_extrinsic,
    build_bev_occupancy
)
from logger.custom_logger import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


class BEVOccupancyDataset(Dataset):
    """
    PyTorch Dataset for nuScenes BEV 2D Occupancy.

    For each sample returns:
        imgs        → (6, 3, IMG_H, IMG_W)
        intrinsics  → (6, 3, 3)
        extrinsics  → (6, 4, 4)
        occ_gt      → (BEV_H, BEV_W)
    """

    def __init__(self,
                 dataroot: str = DATAROOT,
                 version:  str = VERSION):
        try:
            self.nusc    = NuScenes(
                version  = version,
                dataroot = dataroot,
                verbose  = False
            )
            self.samples = self.nusc.sample

            logger.info(
                f"Dataset loaded | "
                f"version: {version} | "
                f"samples: {len(self.samples)}"
            )

        except Exception as e:
            raise BEVException(
                "Failed to initialize BEVOccupancyDataset", e
            ) from e


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> dict:
        try:
            sample     = self.samples[idx]
            imgs       = []
            intrinsics = []
            extrinsics = []

            for cam_name in CAM_NAMES:
                img, K, E = self._load_camera(sample, cam_name)
                imgs.append(img)
                intrinsics.append(K)
                extrinsics.append(E)

            occ_gt = self._load_lidar_bev(sample)

            return {
                'imgs'      : torch.stack(imgs),        # (6,3,H,W)
                'intrinsics': torch.stack(intrinsics),  # (6,3,3)
                'extrinsics': torch.stack(extrinsics),  # (6,4,4)
                'occ_gt'    : occ_gt                    # (200,200)
            }

        except Exception as e:
            raise BEVException(
                f"Failed to load sample at index {idx}", e
            ) from e


    # ──────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────

    def _load_camera(self, sample: dict, cam_name: str):
        cam_token = sample['data'][cam_name]
        cam_data  = self.nusc.get('sample_data', cam_token)
        calib     = self.nusc.get(
            'calibrated_sensor',
            cam_data['calibrated_sensor_token']
        )

        img_path = os.path.join(self.nusc.dataroot, cam_data['filename'])

        img = preprocess_image(img_path)
        K   = preprocess_intrinsic(
                  calib['camera_intrinsic'],
                  orig_w=cam_data['width'],
                  orig_h=cam_data['height']
              )
        E   = preprocess_extrinsic(
                  calib['rotation'],
                  calib['translation']
              )

        return img, K, E


    def _load_lidar_bev(self, sample: dict) -> torch.Tensor:
        """
        Manual NUM_SWEEPS-sweep aggregation in sensor frame.

        WHY MANUAL: nuscenes-devkit 1.1.11 has a bug where
        from_file_multisweep silently returns 1 sweep when
        the prev chain token is broken in mini metadata.

        APPROACH: Walk the prev chain ourselves, load each
        sweep as raw points in sensor frame, concatenate all,
        then pass the merged cloud to build_bev_occupancy
        which handles sensor→ego transform internally.
        """
        lidar_token = sample['data']['LIDAR_TOP']
        ref_data    = self.nusc.get('sample_data', lidar_token)
        ref_calib   = self.nusc.get(
            'calibrated_sensor',
            ref_data['calibrated_sensor_token']
        )

        all_points = []
        current    = ref_data

        for sweep_idx in range(NUM_SWEEPS):
            pc_path = os.path.join(
                self.nusc.dataroot,
                current['filename']
            )
            pc = LidarPointCloud.from_file(pc_path)
            all_points.append(pc.points.copy())          # (4, N)

            if current['prev'] == '':
                logger.debug(                            # ✅ debug not info
                    f"Sweep chain ended at sweep {sweep_idx + 1}"
                )
                break
            current = self.nusc.get('sample_data', current['prev'])

        merged = np.concatenate(all_points, axis=1)     # (4, N_total)

        logger.debug(                                    # ✅ debug not info
            f"Multi-sweep loaded | "
            f"sweeps: {len(all_points)} | "
            f"total points: {merged.shape[1]:,}"
        )

        occ = build_bev_occupancy(
            lidar_points      = merged,
            lidar_rotation    = ref_calib['rotation'],
            lidar_translation = ref_calib['translation']
        )

        return occ


# ──────────────────────────────────────────────────────
# DataLoader factory function
# ──────────────────────────────────────────────────────

def get_dataloaders(
    dataroot: str = DATAROOT,
    version:  str = VERSION
):
    """
    Returns:
        train_loader  → batched DataLoader for training
        val_loader    → batched DataLoader (optional use)
        val_ds        → Subset with .indices (per-sample val loop)
        full_dataset  → full dataset (index access in val loop)

    Colab usage:
        for idx in val_ds.indices:
            s = full_dataset[idx]
    """
    try:
        full_dataset = BEVOccupancyDataset(
            dataroot=dataroot,
            version=version
        )

        total      = len(full_dataset)
        train_size = int(TRAIN_SPLIT * total)
        val_size   = total - train_size

        train_ds, val_ds = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        _pin = torch.cuda.is_available()               # True on Colab GPU

        train_loader = DataLoader(
            train_ds,
            batch_size  = BATCH_SIZE,
            shuffle     = True,
            num_workers = NUM_WORKERS,
            pin_memory  = _pin
        )

        val_loader = DataLoader(
            val_ds,
            batch_size  = BATCH_SIZE,
            shuffle     = False,
            num_workers = NUM_WORKERS,
            pin_memory  = _pin
        )

        logger.info(
            f"DataLoaders ready | "
            f"train: {train_size} | "
            f"val: {val_size} | "
            f"pin_memory: {_pin}"
        )

        return train_loader, val_loader, val_ds, full_dataset

    except Exception as e:
        raise BEVException(
            "Failed to create DataLoaders", e
        ) from e