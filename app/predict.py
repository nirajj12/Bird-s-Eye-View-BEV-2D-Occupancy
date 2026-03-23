# app/predict.py
# ══════════════════════════════════════════════════════
# Model inference for FastAPI backend
# ══════════════════════════════════════════════════════

import os
import sys
import time
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config.config as cfg
from models.bev_model       import BEVOccupancyModel
from data.preprocess        import (
    preprocess_image,
    preprocess_intrinsic,
    preprocess_extrinsic,
    build_bev_occupancy
)
from utils.metrics          import compute_metrics
from nuscenes.nuscenes      import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from logger.custom_logger   import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


class BEVPredictor:
    """
    Main inference class for FastAPI backend.
    """

    def __init__(self):
        self.device = cfg.get_device()
        self.model  = None
        self.nusc   = None
        self._load_model()
        self._load_nuscenes()

    def _load_model(self):
        """Load trained model from checkpoint."""
        try:
            ckpt_path = os.path.join(
                cfg.CKPT_DIR, 'best_model.pth'
            )

            self.model = BEVOccupancyModel(
                pretrained=False
            ).to(self.device)

            if os.path.exists(ckpt_path):
                ckpt = torch.load(
                    ckpt_path,
                    map_location=self.device
                )
                self.model.load_state_dict(
                    ckpt['model_state']
                )
                self.epoch    = ckpt.get('epoch', 0)
                self.best_iou = ckpt.get('val_iou', 0.0)
                logger.info(
                    f"Model loaded | "
                    f"epoch: {self.epoch} | "
                    f"IoU: {self.best_iou:.4f}"
                )
            else:
                logger.warning(
                    "No checkpoint found — "
                    "using untrained model"
                )
                self.epoch    = 0
                self.best_iou = 0.0

            self.model.eval()

        except Exception as e:
            raise BEVException(
                "Failed to load model", e
            ) from e

    def _load_nuscenes(self):
        """Load nuScenes dataset."""
        try:
            self.nusc = NuScenes(
                version  = cfg.VERSION,
                dataroot = cfg.DATAROOT,
                verbose  = False
            )
            logger.info(
                f"nuScenes loaded | "
                f"samples: {len(self.nusc.sample)}"
            )
        except Exception as e:
            raise BEVException(
                "Failed to load nuScenes", e
            ) from e

    def get_all_samples(self) -> list:
        """Return all available samples."""
        samples = []
        for i, sample in enumerate(self.nusc.sample):
            scene = self.nusc.get(
                'scene', sample['scene_token']
            )
            samples.append({
                'index'     : i,
                'token'     : sample['token'],
                'scene_name': scene['name'],
                'timestamp' : sample['timestamp']
            })
        return samples

    def predict_sample(self, sample_idx: int) -> dict:
        """
        Run inference on nuScenes sample.
        Returns BEV prediction + metrics + camera images.
        """
        try:
            start_time = time.time()
            sample     = self.nusc.sample[sample_idx]

            imgs       = []
            intrinsics = []
            extrinsics = []
            cam_images = []

            # Load all 6 cameras
            for cam_name in cfg.CAM_NAMES:
                cam_token = sample['data'][cam_name]
                cam_data  = self.nusc.get(
                    'sample_data', cam_token
                )
                calib = self.nusc.get(
                    'calibrated_sensor',
                    cam_data['calibrated_sensor_token']
                )

                img_path = os.path.join(
                    self.nusc.dataroot,
                    cam_data['filename']
                )

                # Raw image for display
                raw = cv2.imread(img_path)
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                raw = cv2.resize(raw, (352, 128))
                cam_images.append(raw)

                # Preprocessed for model
                img = preprocess_image(img_path)
                K   = preprocess_intrinsic(
                    calib['camera_intrinsic'],
                    cam_data['width'],
                    cam_data['height']
                )
                E   = preprocess_extrinsic(
                    calib['rotation'],
                    calib['translation']
                )
                imgs.append(img)
                intrinsics.append(K)
                extrinsics.append(E)

            # Stack and move to device
            imgs_t = torch.stack(imgs).unsqueeze(0).to(self.device)
            intr_t = torch.stack(intrinsics).unsqueeze(0).to(self.device)
            extr_t = torch.stack(extrinsics).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                output = self.model(imgs_t, intr_t, extr_t)
                pred   = output['occ']

            # Ground truth from LiDAR
            lidar_data  = self.nusc.get(
                'sample_data',
                sample['data']['LIDAR_TOP']
            )
            lidar_calib = self.nusc.get(
                'calibrated_sensor',
                lidar_data['calibrated_sensor_token']
            )
            pc = LidarPointCloud.from_file(
                os.path.join(
                    self.nusc.dataroot,
                    lidar_data['filename']
                )
            )
            gt = build_bev_occupancy(
                pc.points,
                lidar_calib['rotation'],
                lidar_calib['translation']
            ).unsqueeze(0).unsqueeze(0).to(self.device)

            # Metrics
            metrics = compute_metrics(pred, gt)

            # Timing
            inference_time = time.time() - start_time

            # Convert to base64
            bev_img   = self._to_bev_b64(pred[0])
            gt_img    = self._to_bev_b64(gt[0])
            error_img = self._to_error_b64(pred[0], gt[0])
            cam_grid  = self._cams_to_b64(cam_images)

            # Occupancy stats
            pred_np  = pred[0].squeeze().cpu().numpy()
            occ_pct  = float(
                (pred_np >= 0.5).sum()
            ) / pred_np.size * 100

            logger.info(
                f"Sample {sample_idx} | "
                f"IoU: {metrics['occ_iou']:.4f} | "
                f"time: {inference_time:.2f}s"
            )

            return {
                'success'     : True,
                'sample_idx'  : sample_idx,
                'bev_image'   : bev_img,
                'gt_image'    : gt_img,
                'error_image' : error_img,
                'camera_grid' : cam_grid,
                'metrics'     : {
                    'occ_iou'     : round(metrics['occ_iou'], 4),
                    'dwe'         : round(metrics['dwe'], 6),
                    'occupied_pct': round(occ_pct, 1),
                    'free_pct'    : round(100 - occ_pct, 1),
                    'inference_ms': round(inference_time * 1000, 1)
                },
                'model_info'  : {
                    'epoch'   : self.epoch,
                    'best_iou': round(self.best_iou, 4)
                }
            }

        except Exception as e:
            raise BEVException(
                f"Prediction failed for sample {sample_idx}", e
            ) from e

    def _to_bev_b64(self, tensor: torch.Tensor) -> str:
        """Convert BEV tensor to base64 PNG."""
        arr = tensor.squeeze().detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#0b0f16')
        ax.set_facecolor('#0b0f16')
        ax.imshow(
            arr, cmap='hot', origin='lower',
            extent=[
                cfg.X_RANGE[0], cfg.X_RANGE[1],
                cfg.Y_RANGE[0], cfg.Y_RANGE[1]
            ],
            vmin=0, vmax=1
        )
        ax.plot(0, 0, 'b^', markersize=10, label='Ego')
        ax.set_xlabel('X (m)', color='white', fontsize=9)
        ax.set_ylabel('Y (m)', color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        ax.legend(fontsize=8, facecolor='#0b0f16',
                  labelcolor='white')
        plt.tight_layout()
        return self._fig_to_b64(fig)

    def _to_error_b64(self,
                       pred: torch.Tensor,
                       gt:   torch.Tensor) -> str:
        """Convert error map to base64 PNG."""
        pred_np = (
            pred.squeeze().detach().cpu().numpy() >= 0.5
        ).astype(float)
        gt_np   = gt.squeeze().detach().cpu().numpy()
        error   = np.abs(pred_np - gt_np)

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#0b0f16')
        ax.set_facecolor('#0b0f16')
        ax.imshow(
            error, cmap='RdYlGn_r', origin='lower',
            extent=[
                cfg.X_RANGE[0], cfg.X_RANGE[1],
                cfg.Y_RANGE[0], cfg.Y_RANGE[1]
            ],
            vmin=0, vmax=1
        )
        ax.plot(0, 0, 'b^', markersize=10)
        ax.set_xlabel('X (m)', color='white', fontsize=9)
        ax.set_ylabel('Y (m)', color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        plt.tight_layout()
        return self._fig_to_b64(fig)

    def _fig_to_b64(self, fig) -> str:
        """Convert matplotlib figure to base64."""
        buf = io.BytesIO()
        fig.savefig(
            buf, format='png', dpi=100,
            bbox_inches='tight',
            facecolor=fig.get_facecolor()
        )
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(
            buf.read()
        ).decode('utf-8')

    def _cams_to_b64(self, images: list) -> list:
        """Convert camera images to base64 list."""
        result = []
        for img in images:
            pil = Image.fromarray(img)
            buf = io.BytesIO()
            pil.save(buf, format='JPEG', quality=85)
            buf.seek(0)
            result.append(
                base64.b64encode(
                    buf.read()
                ).decode('utf-8')
            )
        return result