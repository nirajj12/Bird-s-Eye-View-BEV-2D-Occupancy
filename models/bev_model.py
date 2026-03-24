# models/bev_model.py
# ══════════════════════════════════════════════════════
# Full BEV 2D Occupancy Model
# Assembles: Backbone → BEVFormerLite → BEVDecoder → OccupancyHead
#
# HACKATHON: Bird's-Eye-View 2D Occupancy
# Input:  6 camera images + camera params
# Output: 2D BEV logit grid (200×200)
#         sigmoid(logits) = P(occupied)
#
# CHANGE LOG vs old version:
#   - LSSViewTransformer → BEVFormerLite (geometry-based)
#   - BEVDecoder: no longer returns (decoded, bev_aux)
#                 returns decoded only
#   - OccupancyHead: signature changed
#                 old: forward(bev_decoded, img_interp) → occ
#                 new: forward(bev_decoded) → (occ_logits, aux_logits)
#   - img_pool removed — was geometrically meaningless
#   - forward() returns (occ_logits, aux_logits), not a dict
#   - compute_loss() updated to match new total_occupancy_loss signature
# ══════════════════════════════════════════════════════

import torch
import torch.nn as nn

from config.config import (
    IMG_CHANNELS, BEV_CHANNELS,
    BEV_H, BEV_W
)
from models.backbone            import ImageBackbone
from models.bev_former_lite     import BEVFormerLite      # ← NEW
from models.bev_decoder         import (
    BEVDecoder,
    OccupancyHead,
    total_occupancy_loss
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
      - BEVDecoder:      2D FCN (FastOcc Eq. 3-5, k×Z speedup)
      - OccupancyHead:   dual-output (main logits + aux logits)

    Input:
        imgs:       (B, 6, 3, H, W)
        intrinsics: (B, 6, 3, 3)       K matrices — scaled to IMG_H×IMG_W
        extrinsics: (B, 6, 4, 4)       T_cam→ego from preprocess_extrinsic

    Output (from forward):
        occ_logits: (B, 1, 200, 200)   raw logits — NOT probabilities
        aux_logits: (B, 1, 200, 200)   auxiliary raw logits
        → apply sigmoid() at inference time to get P(occupied)
    """

    def __init__(self,
                 img_channels: int  = IMG_CHANNELS,   # 128
                 bev_channels: int  = BEV_CHANNELS,   # 64
                 bev_h:        int  = BEV_H,           # 200
                 bev_w:        int  = BEV_W,           # 200
                 pretrained:   bool = True):
        super().__init__()

        try:
            # ── Module 1: Image backbone ──────────────────
            # Shared ResNet50 + FPN across all 6 cameras
            # Output: (B*6, 128, fH, fW) where fH=32, fW=88
            self.backbone = ImageBackbone(
                out_channels = img_channels,
                pretrained   = pretrained
            )

            # ── Module 2: Geometry-based view transformer ─
            # Replaces LSSViewTransformer entirely.
            # Projects each ego-frame BEV cell onto cameras
            # via exact camera geometry (K, R, t).
            # Output: (B, 128, 200, 200)
            self.view_transformer = BEVFormerLite(
                in_channels = img_channels,
                bev_h       = bev_h,
                bev_w       = bev_w
            )

            # ── Module 3: 2D FCN decoder (FastOcc) ────────
            # Replaces 3D conv with 2D conv over channels.
            # Speedup = k × Z (Eq. 5 FastOcc paper).
            # Input:  (B, 128, 200, 200)
            # Output: (B,  64, 200, 200)
            self.bev_decoder = BEVDecoder(
                in_channels  = img_channels,
                out_channels = bev_channels
            )

            # ── Module 4: Dual occupancy head ─────────────
            # Returns raw logits (no sigmoid).
            # Main head + auxiliary head share BEV features.
            # Input:  (B,  64, 200, 200)
            # Output: (B,   1, 200, 200) × 2
            self.occ_head = OccupancyHead(
                bev_channels = bev_channels
            )

            self.bev_h = bev_h
            self.bev_w = bev_w

            # Parameter count logging
            total     = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters()
                           if p.requires_grad)
            logger.info(
                f"BEVOccupancyModel ready | "
                f"total: {total:,} params | "
                f"trainable: {trainable:,}"
            )

        except Exception as e:
            raise BEVException(
                "Failed to init BEVOccupancyModel", e
            ) from e

    def forward(self,
                imgs:       torch.Tensor,
                intrinsics: torch.Tensor,
                extrinsics: torch.Tensor
                ) -> tuple:
        """
        Full forward pass.

        Args:
            imgs:       (B, 6, 3, H, W)
            intrinsics: (B, 6, 3, 3)
            extrinsics: (B, 6, 4, 4)  T_cam→ego

        Returns:
            occ_logits: (B, 1, 200, 200)  ← backprop through this
            aux_logits: (B, 1, 200, 200)  ← aux supervision
        """
        try:
            B, N, C, H, W = imgs.shape

            # ── Step 1: backbone feature extraction ───────
            # Run all 6 cameras through shared backbone in parallel
            imgs_flat = imgs.view(B * N, C, H, W)     # (B*6, 3, H, W)
            feats_flat = self.backbone(imgs_flat)       # (B*6, 128, 32, 88)

            _, Cf, fH, fW = feats_flat.shape
            feats = feats_flat.view(B, N, Cf, fH, fW)  # (B, 6, 128, 32, 88)

            # ── Step 2: geometry-based BEV construction ───
            # Each ego-frame BEV cell samples features from
            # the camera where it is visible via exact projection
            bev_feat = self.view_transformer(
                feats,
                intrinsics,
                extrinsics
            )
            # (B, 128, 200, 200)

            # ── Step 3: 2D FCN decode ─────────────────────
            bev_decoded = self.bev_decoder(bev_feat)
            # (B, 64, 200, 200)

            # ── Step 4: occupancy prediction ──────────────
            # Returns raw logits — sigmoid applied in loss/eval
            occ_logits, aux_logits = self.occ_head(bev_decoded)
            # Both: (B, 1, 200, 200)

            return occ_logits, aux_logits

        except Exception as e:
            raise BEVException(
                "BEVOccupancyModel forward failed", e
            ) from e

    def compute_loss(self,
                 occ_logits: torch.Tensor,
                 aux_logits: torch.Tensor,
                 occ_gt:     torch.Tensor
                 ) -> dict:
        """
        Focal + Dice + Smoothness loss (replaces total_occupancy_loss).

        Args:
            occ_logits: (B, 1, 200, 200)  main head raw logits
            aux_logits: (B, 1, 200, 200)  aux head raw logits
            occ_gt:     (B, 200, 200)     binary GT from LiDAR

        Returns:
            dict with 'total' (call .backward() on this),
            'focal', 'dice', 'smooth', 'aux', 'pw' for logging
        """
        try:
            import torch.nn.functional as F

            gt = occ_gt.unsqueeze(1).float()   # (B, 1, 200, 200)

            # ── Dynamic pos_weight ─────────────────────────────────────────
            n_pos = gt.sum().clamp(min=1.0)
            n_neg = (1.0 - gt).sum().clamp(min=1.0)
            pw    = (n_neg / n_pos).clamp(max=30.0).detach()
            pw_t  = torch.tensor([pw.item()], device=occ_logits.device)

            # ── Focal Loss (main head) ─────────────────────────────────────
            bce   = F.binary_cross_entropy_with_logits(
                        occ_logits, gt,
                        pos_weight=pw_t,
                        reduction='none')
            pt    = torch.exp(-bce)
            focal = (0.75 * (1.0 - pt) ** 2.0 * bce).mean()

            # ── Dice Loss (main head) ──────────────────────────────────────
            prob  = torch.sigmoid(occ_logits)
            inter = (prob * gt).sum()
            dice  = 1.0 - (2.0 * inter + 1.0) / (prob.sum() + gt.sum() + 1.0)

            # ── BEV Smoothness Regularizer ─────────────────────────────────
            sx     = (occ_logits[:, :, :, 1:] - occ_logits[:, :, :, :-1]).abs().mean()
            sy     = (occ_logits[:, :, 1:, :] - occ_logits[:, :, :-1, :]).abs().mean()
            smooth = 0.01 * (sx + sy)

            # ── Auxiliary Head Loss (simple BCE — keeps aux head alive) ────
            aux_loss = F.binary_cross_entropy_with_logits(
                        aux_logits, gt,
                        pos_weight=pw_t)

            # ── Combined ───────────────────────────────────────────────────
            total = focal + 0.5 * dice + smooth + 0.3 * aux_loss

            return {
                'total':  total,
                'focal':  focal.detach().item(),
                'dice':   dice.detach().item(),
                'smooth': smooth.detach().item(),
                'aux':    aux_loss.detach().item(),
                'pw':     pw.item()
            }

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
        Inference-only helper. Returns binary occupancy map.

        Args:
            threshold: P(occupied) cutoff. Default 0.5.
                       Lower → more detections, more false positives.
                       Higher → fewer detections, more false negatives.

        Returns:
            occ_binary: (B, 1, 200, 200) — {0, 1} tensor
        """
        occ_logits, _ = self.forward(imgs, intrinsics, extrinsics)
        return (torch.sigmoid(occ_logits) > threshold).float()