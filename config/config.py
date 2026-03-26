# config/config.py
# ══════════════════════════════════════════════════════
# Central configuration for BEV Occupancy Project
# ALL settings live here — import from here everywhere
# ══════════════════════════════════════════════════════

import os
import torch


# ── Dataset ────────────────────────────────────────────
DATAROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'dataset',
    'nuscenes_data'
)
VERSION  = 'v1.0-mini'


# ── Camera ─────────────────────────────────────────────
CAM_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]
NUM_CAMS = len(CAM_NAMES)   # 6

IMG_H = 256
IMG_W = 704

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


# ── BEV Grid ───────────────────────────────────────────
BEV_H = 200
BEV_W = 200

X_RANGE = (-40.0, 40.0)
Y_RANGE = (-40.0, 40.0)
Z_RANGE = (-1.0,  5.4)

BEV_RES = (X_RANGE[1] - X_RANGE[0]) / BEV_W   # 0.4 m/px


# ── LiDAR Multi-Sweep ──────────────────────────────────
# V2+ uses 5 sweeps for denser GT occupancy
NUM_SWEEPS = 5


# ── Model Architecture ─────────────────────────────────
IMG_CHANNELS = 128
BEV_CHANNELS = 64


# ── Training ───────────────────────────────────────────
BATCH_SIZE   = 2
EPOCHS       = 60       # V3: 60 epochs (was 20)
LR           = 2e-4
WEIGHT_DECAY = 1e-4

TRAIN_SPLIT  = 0.8
NUM_WORKERS  = 0 if not torch.cuda.is_available() else 2
# ↑ 0 on Mac (MPS/CPU), 2 on Colab GPU — auto-detected


# ── Evaluation ─────────────────────────────────────────
# From V2 threshold sweep — 0.80 gives best IoU
THRESHOLD = 0.80


# ── V3 Loss Hyperparameters ────────────────────────────
# Phase 1: Epochs 1 → WARMUP_EPOCHS  (focal only, no Lovász)
# Phase 2: Epochs WARMUP_EPOCHS → PHASE2_START  (full loss)
# Phase 3: Epochs PHASE2_START → EPOCHS  (DWE-heavy)

WARMUP_EPOCHS  = 5     # Lovász OFF before this epoch
PHASE2_START   = 40    # Switch to DWE-heavy weights after this

# Loss component weights — Phase 1 (IoU focus)
LOVASZ_WEIGHT  = 1.0
DWE_WEIGHT_P1  = 0.25
CONF_WEIGHT_P1 = 0.20
TV_WEIGHT_P1   = 0.05
AUX_WEIGHT     = 0.30  # aux BEV head loss weight (always constant)

# Loss component weights — Phase 2 (DWE focus)
DWE_WEIGHT_P2  = 0.40
CONF_WEIGHT_P2 = 0.50
TV_WEIGHT_P2   = 0.15

# Focal loss params
FOCAL_ALPHA    = 0.75
FOCAL_GAMMA    = 2.0

# Dynamic pos_weight cap (prevents instability on sparse GT)
POS_WEIGHT_CAP = 5.0

# Spatial pos_weight bounds
NEAR_POS_WEIGHT = 2.0  # lower near ego → fewer FP near center
FAR_POS_WEIGHT  = 6.0  # higher far field → better recall at edges

# DWE spatial weight sigma (for smooth exponential decay)
DWE_SIGMA      = 30.0
DWE_NEAR_BOOST = 3.0

# Gradient clipping
GRAD_CLIP      = 1.0


# ── Checkpoint Paths ───────────────────────────────────
CKPT_DIR    = 'checkpoints'
RESULTS_DIR = 'results'
LOGS_DIR    = 'logs'

# Drive paths (used in Colab only)
DRIVE_CKPT_DIR = '/content/drive/MyDrive/BEV_PROJECT'
V2_CKPT_NAME   = 'bestmodel.pth'      # your existing V2 checkpoint
V3_CKPT_NAME   = 'best_v3.pth'        # V3 will save here

V2_CKPT_PATH   = os.path.join(DRIVE_CKPT_DIR, V2_CKPT_NAME)
V3_CKPT_PATH   = os.path.join(DRIVE_CKPT_DIR, V3_CKPT_NAME)


os.makedirs(CKPT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)


# ── Device ─────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()


# ── Sanity Print (optional) ────────────────────────────
if __name__ == '__main__':
    print(f"Device      : {DEVICE}")
    print(f"Dataroot    : {DATAROOT}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Threshold   : {THRESHOLD}")
    print(f"Warmup ends : epoch {WARMUP_EPOCHS}")
    print(f"Phase2 start: epoch {PHASE2_START}")
    print(f"Num workers : {NUM_WORKERS}")
    print(f"V3 ckpt     : {V3_CKPT_PATH}")