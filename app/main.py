# app/main.py
# ══════════════════════════════════════════════════════
# FastAPI server for BEV Occupancy Predictor
# ══════════════════════════════════════════════════════

import os
import sys
import shutil
import tempfile

from fastapi import (
    FastAPI, UploadFile, File,
    Form, HTTPException, Request
)
from fastapi.responses  import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating  import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.predict          import BEVPredictor
from app.video_processor  import VideoProcessor
from logger.custom_logger import CustomLogger

logger    = CustomLogger().get_logger(__name__)
predictor = None

app = FastAPI(
    title       = "BEV Occupancy Predictor",
    description = "MAHE Mobility Hackathon 2026 — PS3",
    version     = "1.0"
)

# ── Middleware ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
    allow_credentials = True
)

# ── Static + Templates ──────────────────────────────
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)
templates = Jinja2Templates(directory="templates")


# ── Startup ─────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global predictor
    logger.info("Loading BEV model...")
    predictor = BEVPredictor()
    predictor.video_processor = VideoProcessor(
        model  = predictor.model,
        device = predictor.device
    )
    logger.info("All systems ready!")


# ── Routes ──────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve main UI."""
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )


@app.get("/health")
def health():
    """Health check + model info."""
    return {
        "status"   : "ok",
        "model"    : "BEVOccupancyModel",
        "epoch"    : predictor.epoch    if predictor else 0,
        "best_iou" : predictor.best_iou if predictor else 0.0,
        "device"   : str(predictor.device) if predictor else "cpu"
    }


@app.get("/samples")
def get_samples():
    """Get all available nuScenes samples."""
    try:
        samples = predictor.get_all_samples()
        return {
            "samples": samples,
            "total"  : len(samples)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=str(e)
        )


@app.post("/predict/sample")
async def predict_sample(
    sample_idx: int = Form(...)
):
    """
    Predict BEV for a nuScenes sample.
    Returns: BEV heatmap + GT + error + metrics
    """
    try:
        result = predictor.predict_sample(sample_idx)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=str(e)
        )


@app.post("/predict/video")
async def predict_video(
    video: UploadFile = File(...)
):
    """
    Predict BEV from uploaded video.
    Returns: BEV frames + side-by-side + GIF
    """
    try:
        # Save uploaded video to temp file
        suffix = os.path.splitext(video.filename)[1]
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            shutil.copyfileobj(video.file, tmp)
            tmp_path = tmp.name

        # Process video
        result = predictor.video_processor.process_video(
            video_path = tmp_path,
            max_frames = 30
        )

        # Cleanup temp file
        os.unlink(tmp_path)

        return JSONResponse(content={
            "success"      : True,
            "frames"       : result['frames'],
            "side_by_side" : result['side_by_side'],
            "gif_b64"      : result['gif_b64'],
            "total"        : result['total']
        })

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=str(e)
        )