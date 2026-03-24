# models/bev_model.py
# ══════════════════════════════════════════════════════
# Full BEV 2D Occupancy Model
# V3: compute_loss() passes epoch → enables DWE phases
# Architecture unchanged
# ══════════════════════════════════════════════════════

import torch
import torch.nn as nn

from config.config import (
    IMG_CHANNELS, BEV_CHANNELS,
    BEV_H, BEV_W
)
from models.backbone            import ImageBackbone
from models.bev_former_lite     import BEVFormerLite
from models.bev_decoder         import (
    BEVDecoder,
    OccupancyHead,
    total_occupancy_loss          # ✅ V3 version with DWE
)
from logger.custom_logger       import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


class BEVOccupancyModel(nn.Module):
    """
    End-to-end BEV 2D Occupancy Model.

    Architecture:
      - ImageBackbone:   shared ResNet50+FPN across all 6 cameras
      - BEVFormerLite:   geometry projection (Delving paper Eq. 5)
      - BEVDecoder:      2D FCN (FastOcc Eq. 3-5)
      - OccupancyHead:   dual-output (main logits + aux logits)

    Input:
        imgs:       (B, 6, 3, H, W)
        intrinsics: (B, 6, 3, 3)
        extrinsics: (B, 6, 4, 4)  T_cam→ego

    Output:
        occ_logits: (B, 1, 200, 200)  raw logits
        aux_logits: (B, 1, 200, 200)  auxiliary raw logits
    """

    def __init__(self,
                 img_channels: int  = IMG_CHANNELS,
                 bev_channels: int  = BEV_CHANNELS,
                 bev_h:        int  = BEV_H,
                 bev_w:        int  = BEV_W,
                 pretrained:   bool = True):
        super().__init__()

        try:
            self.backbone = ImageBackbone(
                out_channels = img_channels,
                pretrained   = pretrained
            )

            self.view_transformer = BEVFormerLite(
                in_channels = img_channels,
                bev_h       = bev_h,
                bev_w       = bev_w
            )

            self.bev_decoder = BEVDecoder(
                in_channels  = img_channels,
                out_channels = bev_channels
            )

            self.occ_head = OccupancyHead(
                bev_channels = bev_channels
            )

            self.bev_h = bev_h
            self.bev_w = bev_w

            total     = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters()
                            if p.requires_grad)
            logger.info(
                f"BEVOccupancyModel ready | "
                f"total: {total:,} | trainable: {trainable:,}"
            )

        except Exception as e:
            raise BEVException(
                "Failed to init BEVOccupancyModel", e
            ) from e


    def forward(self,
                imgs:       torch.Tensor,
                intrinsics: torch.Tensor,
                extrinsics: torch.Tensor) -> tuple:
        """
        Args:
            imgs:       (B, 6, 3, H, W)
            intrinsics: (B, 6, 3, 3)
            extrinsics: (B, 6, 4, 4)

        Returns:
            occ_logits: (B, 1, 200, 200)
            aux_logits: (B, 1, 200, 200)
        """
        try:
            B, N, C, H, W = imgs.shape

            imgs_flat  = imgs.view(B * N, C, H, W)
            feats_flat = self.backbone(imgs_flat)

            _, Cf, fH, fW = feats_flat.shape
            feats = feats_flat.view(B, N, Cf, fH, fW)

            bev_feat    = self.view_transformer(feats, intrinsics, extrinsics)
            bev_decoded = self.bev_decoder(bev_feat)
            occ_logits, aux_logits = self.occ_head(bev_decoded)

            return occ_logits, aux_logits

        except Exception as e:
            raise BEVException(
                "BEVOccupancyModel forward failed", e
            ) from e


    def compute_loss(self,
                     occ_logits: torch.Tensor,
                     aux_logits: torch.Tensor,
                     occ_gt:     torch.Tensor,
                     epoch:      int = 1        # ✅ NEW: required for DWE phases
                     ) -> dict:
        """
        Calls total_occupancy_loss from bev_decoder.py.
        Passes epoch for warmup / phase1 / phase2 control.

        Args:
            occ_logits: (B, 1, 200, 200)
            aux_logits: (B, 1, 200, 200)
            occ_gt:     (B, 200, 200)
            epoch:      current training epoch (1-indexed)

        Returns:
            dict with 'total', 'focal', 'dice', 'aux',
                      'dwe', 'conf', 'tv', 'phase'
        """
        try:
            gt = occ_gt.unsqueeze(1).float()   # (B,1,200,200)

            return total_occupancy_loss(
                occ_logits = occ_logits,
                gt         = gt,
                epoch      = epoch,             # ✅ DWE phase control
                aux_logits = aux_logits,
                focal_w    = 1.0,
                dice_w     = 1.0,
                aux_w      = 0.3
            )

        except Exception as e:
            raise BEVException("Loss computation failed", e) from e


    @torch.no_grad()
    def predict(self,
                imgs:       torch.Tensor,
                intrinsics: torch.Tensor,
                extrinsics: torch.Tensor,
                threshold:  float = 0.5
                ) -> torch.Tensor:
        """
        Inference-only. Returns binary occupancy map.

        Returns:
            (B, 1, 200, 200) — {0, 1} tensor
        """
        occ_logits, _ = self.forward(imgs, intrinsics, extrinsics)
        return (torch.sigmoid(occ_logits) > threshold).float()