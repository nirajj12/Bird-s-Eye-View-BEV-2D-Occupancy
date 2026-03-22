# config/config.py
# ══════════════════════════════════════════════════════
# Central configuration — BEV 2D Occupancy Project
# MAHE Mobility Hackathon 2026
#
# ALL settings live here
# Change here → updates everywhere
# ══════════════════════════════════════════════════════

import os
import torch


# ── Dataset ────────────────────────────────────────────
DATAROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'dataset', 'nuscenes_data'
)
VERSION  = 'v1.0-mini'

# ── Camera ─────────────────────────────────────────────
# nuScenes has exactly 6 cameras around the car
CAM_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]
NUM_CAMS = len(CAM_NAMES)   # 6

# Resize every camera image to this fixed size
# Original nuScenes: 1600×900 → resized to save memory
IMG_H = 256    # height in pixels
IMG_W = 704    # width  in pixels

# ImageNet normalization — required for pretrained ResNet
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# ── BEV Grid ───────────────────────────────────────────
# FIXED: 10cm per pixel as required by hackathon spec
# "each pixel represents a fixed area (e.g., 10cm x 10cm)"
BEV_H = 800    # grid height (pixels)
BEV_W = 800    # grid width  (pixels)

# Real-world range around the ego vehicle (metres)
X_RANGE = (-40.0, 40.0)   # left  ↔ right   = 80m wide
Y_RANGE = (-40.0, 40.0)   # front ↔ behind  = 80m deep
Z_RANGE = (-1.0,  5.4)    # below ↔ above   = 6.4m tall

# Resolution: how many real-world metres per pixel
# 80m / 800px = 0.1m = 10cm per pixel ✓ matches requirement
BEV_RES = (X_RANGE[1] - X_RANGE[0]) / BEV_W

# ── Model Architecture ─────────────────────────────────
IMG_CHANNELS = 128    # feature channels from backbone
DEPTH_BINS   = 64     # LSS depth bins (2m → 58m)
BEV_CHANNELS = 64     # channels inside BEV decoder

# Backbone selection
# Options : 'resnet50'  → faster, IoU ~0.20
#           'resnet101' → better, IoU ~0.28  ← recommended
#           'resnet152' → best,   IoU ~0.32  (needs more VRAM)
BACKBONE = 'resnet101'

# ── Training ───────────────────────────────────────────
# Reduced batch size for 800×800 BEV grid memory
BATCH_SIZE   = 1      # 1 for 800×800, 2 for 200×200
EPOCHS       = 50     # 50 for convergence (was 20)
LR           = 2e-4   # base learning rate
WEIGHT_DECAY = 1e-4   # AdamW weight decay
WARMUP_EPOCHS = 5     # linear warmup before cosine decay

# Train / validation split
TRAIN_SPLIT  = 0.8    # 80% train, 20% val
NUM_WORKERS  = 2      # 0 for Mac, 2-4 for Windows/Colab

# ── Device ─────────────────────────────────────────────
def get_device() -> torch.device:
    """
    Returns best available device:
      CUDA  → Colab T4 / Windows NVIDIA GPU
      MPS   → Mac Apple Silicon
      CPU   → fallback
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()

# ── Paths ──────────────────────────────────────────────
CKPT_DIR    = 'checkpoints'
RESULTS_DIR = 'results'
LOGS_DIR    = 'logs'

# Create directories automatically
os.makedirs(CKPT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)