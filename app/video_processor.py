# app/video_processor.py
# ══════════════════════════════════════════════════════
# Video processing for BEV Occupancy Predictor
# Handles: frame extraction, BEV prediction per frame,
#          GIF generation, video output
# ══════════════════════════════════════════════════════

import os
import sys
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import tempfile
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config.config as cfg
from logger.custom_logger   import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


class VideoProcessor:
    """
    Handles video input for BEV prediction.

    Features:
        1. Extract frames from video
        2. Predict BEV for each frame
        3. Generate animated GIF
        4. Side-by-side comparison video
        5. Return frames as base64 for frontend
    """

    def __init__(self, model, device):
        """
        Args:
            model:  loaded BEVOccupancyModel
            device: torch device
        """
        self.model  = model
        self.device = device

        # Default camera params for single camera input
        # (when we don't have nuScenes calibration)
        self.default_K = self._build_default_K()
        self.default_E = self._build_default_E()

        logger.info("VideoProcessor initialized")

    def _build_default_K(self) -> torch.Tensor:
        """
        Build default intrinsic matrix K.
        Uses typical dashcam focal length.
        """
        K = torch.zeros(1, 6, 3, 3)
        K[:, :, 0, 0] = 800.0   # fx
        K[:, :, 1, 1] = 800.0   # fy
        K[:, :, 0, 2] = cfg.IMG_W / 2   # cx
        K[:, :, 1, 2] = cfg.IMG_H / 2   # cy
        K[:, :, 2, 2] = 1.0
        return K.to(self.device)

    def _build_default_E(self) -> torch.Tensor:
        """
        Build default extrinsic matrices.
        Places all 6 virtual cameras around the car.
        """
        E = torch.eye(4).unsqueeze(0).unsqueeze(0)
        E = E.expand(1, 6, 4, 4).contiguous()
        return E.to(self.device)

    def extract_frames(self,
                       video_path: str,
                       max_frames: int = 30,
                       skip_frames: int = 2
                       ) -> list:
        """
        Extract frames from video file.

        Args:
            video_path  : path to video file
            max_frames  : maximum frames to extract
            skip_frames : take every Nth frame

        Returns:
            list of numpy arrays (BGR frames)
        """
        try:
            cap    = cv2.VideoCapture(video_path)
            frames = []
            count  = 0
            total  = 0

            if not cap.isOpened():
                raise ValueError(
                    f"Cannot open video: {video_path}"
                )

            fps    = cap.get(cv2.CAP_PROP_FPS)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            logger.info(
                f"Video info | "
                f"fps: {fps:.1f} | "
                f"size: {width}×{height} | "
                f"total frames: {total_frames}"
            )

            while cap.isOpened() and count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames for speed
                if total % skip_frames == 0:
                    frames.append(frame)
                    count += 1

                total += 1

            cap.release()

            logger.info(
                f"Extracted {len(frames)} frames "
                f"from {total} total"
            )
            return frames

        except Exception as e:
            raise BEVException(
                "Failed to extract frames", e
            ) from e

    def frame_to_tensor(self,
                        frame: np.ndarray
                        ) -> torch.Tensor:
        """
        Convert single video frame to model input tensor.

        Replicates frame across all 6 virtual cameras.
        This is a simplified approach for video input
        (real approach would need 6 synchronized cameras).

        Args:
            frame: BGR numpy array from cv2

        Returns:
            tensor: (1, 6, 3, IMG_H, IMG_W)
        """
        try:
            # BGR → RGB
            frame_rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            # Resize to model input size
            frame_resized = cv2.resize(
                frame_rgb,
                (cfg.IMG_W, cfg.IMG_H),
                interpolation=cv2.INTER_LINEAR
            )

            # Normalize with ImageNet stats
            mean = np.array(cfg.IMG_MEAN, dtype=np.float32)
            std  = np.array(cfg.IMG_STD,  dtype=np.float32)
            img  = frame_resized.astype(np.float32) / 255.0
            img  = (img - mean) / std

            # HWC → CHW
            img = img.transpose(2, 0, 1)
            img_t = torch.tensor(
                img, dtype=torch.float32
            ).unsqueeze(0)  # (1, 3, H, W)

            # Replicate for 6 cameras
            imgs_t = img_t.unsqueeze(1).repeat(
                1, 6, 1, 1, 1
            )  # (1, 6, 3, H, W)

            return imgs_t.to(self.device)

        except Exception as e:
            raise BEVException(
                "Failed to convert frame to tensor", e
            ) from e

    def predict_frame(self,
                      frame: np.ndarray
                      ) -> tuple:
        """
        Run BEV prediction on a single frame.

        Args:
            frame: BGR numpy array

        Returns:
            tuple: (pred_tensor, bev_base64_string)
        """
        try:
            imgs_t = self.frame_to_tensor(frame)

            with torch.no_grad():
                output = self.model(
                    imgs_t,
                    self.default_K,
                    self.default_E
                )
                pred = output['occ']

            bev_b64 = self._pred_to_base64(pred[0])
            return pred, bev_b64

        except Exception as e:
            raise BEVException(
                "Frame prediction failed", e
            ) from e

    def process_video(self,
                      video_path: str,
                      max_frames: int = 30
                      ) -> dict:
        """
        Full video processing pipeline.

        Args:
            video_path: path to video file
            max_frames: max frames to process

        Returns:
            dict with:
                frames      : list of BEV base64 images
                side_by_side: list of side-by-side base64
                gif_b64     : animated GIF as base64
                total       : number of frames processed
        """
        try:
            logger.info(
                f"Processing video: {video_path}"
            )

            # Extract frames
            frames = self.extract_frames(
                video_path,
                max_frames  = max_frames,
                skip_frames = 2
            )

            bev_frames     = []
            side_by_side   = []
            pil_frames     = []   # for GIF

            for i, frame in enumerate(frames):
                logger.info(
                    f"Processing frame {i+1}/{len(frames)}"
                )

                # Predict BEV
                pred, bev_b64 = self.predict_frame(frame)
                bev_frames.append(bev_b64)

                # Create side-by-side image
                sbs = self._make_side_by_side(frame, pred)
                side_by_side.append(sbs['b64'])
                pil_frames.append(sbs['pil'])

            # Generate animated GIF
            gif_b64 = self._make_gif(pil_frames)

            logger.info(
                f"Video processing complete | "
                f"frames: {len(bev_frames)}"
            )

            return {
                'frames'      : bev_frames,
                'side_by_side': side_by_side,
                'gif_b64'     : gif_b64,
                'total'       : len(bev_frames)
            }

        except Exception as e:
            raise BEVException(
                "Video processing pipeline failed", e
            ) from e

    def _pred_to_base64(self,
                        pred: torch.Tensor
                        ) -> str:
        """Convert prediction tensor to base64 PNG."""
        arr = pred.squeeze().detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(
            arr,
            cmap    = 'hot',
            origin  = 'lower',
            extent  = [
                cfg.X_RANGE[0], cfg.X_RANGE[1],
                cfg.Y_RANGE[0], cfg.Y_RANGE[1]
            ],
            vmin=0, vmax=1
        )
        ax.plot(0, 0, 'b^', markersize=8)
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.tick_params(labelsize=7)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80,
                    bbox_inches='tight')
        plt.close()
        buf.seek(0)

        return base64.b64encode(
            buf.read()
        ).decode('utf-8')

    def _make_side_by_side(self,
                            frame: np.ndarray,
                            pred:  torch.Tensor
                            ) -> dict:
        """
        Create side-by-side image:
        Left: original camera frame
        Right: BEV prediction heatmap

        Returns:
            dict with 'b64' (base64) and 'pil' (PIL Image)
        """
        try:
            # Original frame
            frame_rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )
            frame_resized = cv2.resize(
                frame_rgb, (400, 225)
            )

            # BEV prediction
            arr = pred[0].squeeze().detach().cpu().numpy()

            fig, axes = plt.subplots(
                1, 2, figsize=(10, 4)
            )
            fig.patch.set_facecolor('#0b0f16')

            # Camera frame
            axes[0].imshow(frame_resized)
            axes[0].set_title(
                'Camera Input',
                color='white', fontsize=10
            )
            axes[0].axis('off')

            # BEV heatmap
            axes[1].imshow(
                arr,
                cmap   = 'hot',
                origin = 'lower',
                extent = [
                    cfg.X_RANGE[0], cfg.X_RANGE[1],
                    cfg.Y_RANGE[0], cfg.Y_RANGE[1]
                ],
                vmin=0, vmax=1
            )
            axes[1].plot(0, 0, 'b^', markersize=8)
            axes[1].set_title(
                'BEV Occupancy',
                color='white', fontsize=10
            )
            axes[1].set_facecolor('#0b0f16')
            axes[1].tick_params(
                colors='white', labelsize=7
            )
            axes[1].set_xlabel('X (m)', color='white')
            axes[1].set_ylabel('Y (m)', color='white')

            plt.tight_layout()

            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(
                buf, format='png', dpi=80,
                bbox_inches='tight',
                facecolor='#0b0f16'
            )
            plt.close()
            buf.seek(0)

            # PIL image for GIF
            pil_img = Image.open(buf).copy()
            buf.seek(0)

            b64 = base64.b64encode(
                buf.read()
            ).decode('utf-8')

            return {'b64': b64, 'pil': pil_img}

        except Exception as e:
            raise BEVException(
                "Failed to create side-by-side", e
            ) from e

    def _make_gif(self,
                  pil_frames: list,
                  duration:   int = 200
                  ) -> str:
        """
        Create animated GIF from PIL frames.

        Args:
            pil_frames: list of PIL Images
            duration:   ms per frame

        Returns:
            base64 encoded GIF string
        """
        try:
            if not pil_frames:
                return ""

            buf = io.BytesIO()

            pil_frames[0].save(
                buf,
                format     = 'GIF',
                save_all   = True,
                append_images = pil_frames[1:],
                duration   = duration,
                loop       = 0   # loop forever
            )

            buf.seek(0)
            gif_b64 = base64.b64encode(
                buf.read()
            ).decode('utf-8')

            logger.info(
                f"GIF created | frames: {len(pil_frames)}"
            )
            return gif_b64

        except Exception as e:
            raise BEVException(
                "Failed to create GIF", e
            ) from e