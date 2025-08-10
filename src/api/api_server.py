"""
MIDI Generator API Server

FastAPI-based REST API for converting audio stems to MIDI files.
"""

import os
import tempfile
import uuid
from pathlib import Path
import json
from typing import Optional, List
import asyncio
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Resilient imports to support both package and module execution
try:
    # Preferred package-style imports (run via: uv run python -m src.cli ...)
    from ..inference.inference import create_inference_engine, InferenceConfig
    from ..utils.logging import MIDIGeneratorLogger
except Exception as _rel_imp_err:  # pragma: no cover - fallback path
    try:
        # Allow absolute package path when PYTHONPATH includes project root
        from src.inference.inference import create_inference_engine, InferenceConfig
        from src.utils.logging import MIDIGeneratorLogger
    except Exception as _abs_imp_err:
        _msg = (
            " attempted relative import with no known parent package\n\n"
            "How to fix:\n"
            "  1) Run as a module from project root:\n"
            "     uv run python -m src.cli api --port 8000\n"
            "     uv run python -m src.cli transcribe <audio.wav> <model.pth>\n"
            "  2) Or set PYTHONPATH to project root, e.g.:\n"
            "     export PYTHONPATH=.$PYTHONPATH\n\n"
            "If this persists, ensure the repository layout is intact and src/ is a package."
        )
        raise ImportError(_msg) from _abs_imp_err


# Pydantic models for API
class TranscriptionRequest(BaseModel):
    """Request model for transcription."""
    onset_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Onset detection threshold")
    frame_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Frame detection threshold")
    velocity_scale: float = Field(default=127.0, ge=1.0, le=127.0, description="Velocity scaling factor")
    min_note_duration: float = Field(default=0.05, ge=0.01, le=1.0, description="Minimum note duration in seconds")
    clip_length: float = Field(default=4.0, ge=1.0, le=10.0, description="Audio clip length for processing")
    overlap: float = Field(default=0.5, ge=0.0, le=0.9, description="Overlap between clips")


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    job_id: str
    status: str
    message: str
    midi_url: Optional[str] = None
    processing_time: Optional[float] = None
    num_notes: Optional[int] = None
    created_at: str


class JobStatus(BaseModel):
    """Job status model."""
    job_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    message: str
    progress: Optional[float] = None
    midi_url: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


# Global variables
app = FastAPI(
    title="MIDI Generator API",
    description="Convert audio stems to MIDI files using AI transcription",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
inference_engine = None
job_storage = {}  # In-memory job storage (use Redis/DB in production)
logger = MIDIGeneratorLogger("api_server", log_file="api_server.log")
PROFILES = {}

# Configuration
API_CONFIG = {
    "model_path": None,  # Set via environment or startup
    "upload_dir": "uploads",
    "output_dir": "outputs",
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "allowed_extensions": {".wav", ".mp3", ".flac", ".m4a", ".aac"},
    "cleanup_after_hours": 24
}


@app.on_event("startup")
async def startup_event():
    """Initialize the API server."""
    global inference_engine
    global PROFILES
    
    logger.info("🚀 Starting MIDI Generator API Server...")
    
    # Get model path from environment
    model_path = os.getenv("MIDI_MODEL_PATH")
    if not model_path:
        # Look for latest checkpoint
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            patterns = ["*.pth", "*.pt", "*.ckpt"]
            checkpoint_files = []
            for pattern in patterns:
                checkpoint_files.extend(checkpoints_dir.rglob(pattern))
            if checkpoint_files:
                model_path = str(sorted(checkpoint_files)[-1])  # Latest checkpoint
                logger.info(f"📂 Using latest checkpoint: {model_path}")
            else:
                logger.warning("No checkpoint found in 'checkpoints/' (looked for *.pth, *.pt, *.ckpt)")
    
    if not model_path or not Path(model_path).exists():
        logger.error("❌ No model checkpoint found! Set MIDI_MODEL_PATH environment variable.")
        raise RuntimeError("Model checkpoint not found")
    
    API_CONFIG["model_path"] = model_path
    
    # Create directories
    Path(API_CONFIG["upload_dir"]).mkdir(parents=True, exist_ok=True)
    Path(API_CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Load decoder profiles if available
    try:
        profiles_path = Path(__file__).resolve().parent.parent / "inference" / "profiles.json"
        if profiles_path.exists():
            PROFILES = json.loads(profiles_path.read_text())
            logger.info(f" Loaded {len(PROFILES)} decoder profile(s) from {profiles_path}")
        else:
            logger.info(" No decoder profiles found; using API defaults")
    except Exception as e:
        logger.warning(f" Failed to load profiles.json: {e}")
    
    # Initialize inference engine
    try:
        inference_engine = create_inference_engine(model_path)
        logger.success("✅ Inference engine initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize inference engine: {str(e)}")
        raise
    
    logger.success("🎵 MIDI Generator API Server ready!")


@app.get("/", response_model=dict)
async def root():
    """API root endpoint."""
    return {
        "message": "MIDI Generator API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "transcribe": "/transcribe",
            "status": "/status/{job_id}",
            "download": "/download/{job_id}",
            "health": "/health"
        }
    }


@app.get("/health", response_model=dict)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "inference_engine": "ready" if inference_engine else "not_ready",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    profile: Optional[str] = Query(None, description="Decoding profile to apply (e.g., 'bass', 'piano')"),
    onset_threshold: float = Query(0.03, ge=0.0, le=1.0, description="Onset detection threshold"),
    frame_threshold: float = Query(0.08, ge=0.0, le=1.0, description="Frame detection threshold"),
    frame_threshold_off: float = Query(0.04, ge=0.0, le=1.0, description="Frame off-threshold (hysteresis)"),
    velocity_scale: float = Query(127.0, ge=1.0, le=127.0, description="Velocity scaling factor"),
    min_note_duration: float = Query(0.02, ge=0.005, le=1.0, description="Minimum note duration"),
    clip_length: float = Query(4.0, ge=1.0, le=10.0, description="Audio clip length"),
    overlap: float = Query(0.5, ge=0.0, le=0.9, description="Overlap between clips"),
    # Evaluator-style decoding knobs
    smooth_window: int = Query(5, ge=1, le=51, description="Frame prob smoothing window (odd)"),
    min_ioi: float = Query(0.05, ge=0.0, le=1.0, description="Minimum inter-onset interval (s)"),
    merge_gap: float = Query(0.03, ge=0.0, le=1.0, description="Merge gap for same-pitch notes (s)"),
    pitch_min: Optional[int] = Query(None, ge=0, le=127, description="Minimum MIDI pitch to decode"),
    pitch_max: Optional[int] = Query(None, ge=0, le=127, description="Maximum MIDI pitch to decode"),
    enable_fallback_segmentation: bool = Query(False, description="Enable frame-based fallback segmentation")
):
    """
    Transcribe audio file to MIDI.
    
    Upload an audio file and get back a MIDI transcription.
    The process is asynchronous - use the job_id to check status and download results.
    """
    try:
        # Validate file
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = Path(audio_file.filename).suffix.lower()
        if file_ext not in API_CONFIG["allowed_extensions"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {list(API_CONFIG['allowed_extensions'])}"
            )
        
        # Check file size
        content = await audio_file.read()
        if len(content) > API_CONFIG["max_file_size"]:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        upload_path = Path(API_CONFIG["upload_dir"]) / f"{job_id}{file_ext}"
        with open(upload_path, "wb") as f:
            f.write(content)
        
        # Build base config (apply profile defaults if provided)
        if profile is not None:
            if profile not in PROFILES:
                raise HTTPException(status_code=400, detail=f"Unknown profile '{profile}'. Available: {list(PROFILES.keys())}")
            base_cfg = dict(PROFILES[profile])
            logger.info(f" Applying decoder profile: {profile}")
        else:
            base_cfg = {}

        # Always overlay request params on top of base config
        req_cfg = {
            "onset_threshold": onset_threshold,
            "frame_threshold": frame_threshold,
            "frame_threshold_off": frame_threshold_off,
            "velocity_scale": velocity_scale,
            "min_note_duration": min_note_duration,
            "clip_length": clip_length,
            "overlap": overlap,
            # evaluator knobs
            "smooth_window": smooth_window,
            "min_ioi": min_ioi,
            "merge_gap": merge_gap,
            "pitch_min": pitch_min,
            "pitch_max": pitch_max,
            "enable_fallback_segmentation": enable_fallback_segmentation,
        }
        cfg = {**base_cfg, **{k: v for k, v in req_cfg.items() if v is not None}}

        # Create job record
        job_record = {
            "job_id": job_id,
            "status": "pending",
            "message": "Job queued for processing",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "upload_path": str(upload_path),
            "filename": audio_file.filename,
            "config": cfg,
            "profile": profile,
        }
        
        job_storage[job_id] = job_record
        
        # Start background processing
        background_tasks.add_task(process_transcription, job_id)
        
        logger.info(f"🎵 New transcription job created: {job_id}")
        
        return TranscriptionResponse(
            job_id=job_id,
            status="pending",
            message="Transcription job queued successfully",
            created_at=job_record["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Transcription request failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a transcription job."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    return JobStatus(**job)


@app.get("/download/{job_id}")
async def download_midi(job_id: str):
    """Download the generated MIDI file."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    midi_path = Path(API_CONFIG["output_dir"]) / f"{job_id}.mid"
    if not midi_path.exists():
        raise HTTPException(status_code=404, detail="MIDI file not found")
    
    return FileResponse(
        path=str(midi_path),
        filename=f"{Path(job['filename']).stem}.mid",
        media_type="audio/midi"
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    
    # Clean up files
    upload_path = Path(job.get("upload_path", ""))
    if upload_path.exists():
        upload_path.unlink()
    
    midi_path = Path(API_CONFIG["output_dir"]) / f"{job_id}.mid"
    if midi_path.exists():
        midi_path.unlink()
    
    # Remove from storage
    del job_storage[job_id]
    
    return {"message": "Job deleted successfully"}


@app.get("/jobs", response_model=List[JobStatus])
async def list_jobs(status: Optional[str] = Query(None, description="Filter by status")):
    """List all jobs, optionally filtered by status."""
    jobs = list(job_storage.values())
    
    if status:
        jobs = [job for job in jobs if job["status"] == status]
    
    return [JobStatus(**job) for job in jobs]


def process_transcription(job_id: str):
    """Background task to process transcription."""
    try:
        job = job_storage[job_id]
        
        # Update status
        job["status"] = "processing"
        job["message"] = "Processing audio transcription..."
        job["progress"] = 0.1
        
        logger.info(f"🔄 Processing transcription job: {job_id}")
        
        # Get configuration
        config = job["config"]
        upload_path = job["upload_path"]
        try:
            size = os.path.getsize(upload_path)
            logger.info(f" Upload size: {size} bytes -> {upload_path}")
        except Exception:
            logger.warning(f" Could not stat upload file: {upload_path}")
        
        # Update inference engine config
        inference_engine.config.onset_threshold = config["onset_threshold"]
        inference_engine.config.frame_threshold = config["frame_threshold"]
        if "frame_threshold_off" in config:
            inference_engine.config.frame_threshold_off = config["frame_threshold_off"]
        inference_engine.config.velocity_scale = config["velocity_scale"]
        inference_engine.config.min_note_duration = config["min_note_duration"]
        inference_engine.config.clip_length = config["clip_length"]
        inference_engine.config.overlap = config["overlap"]
        # Evaluator-style decoding knobs (optional)
        if "smooth_window" in config and config["smooth_window"]:
            inference_engine.config.smooth_window = int(config["smooth_window"])
        if "min_ioi" in config and config["min_ioi"] is not None:
            inference_engine.config.min_ioi = float(config["min_ioi"])
        if "merge_gap" in config and config["merge_gap"] is not None:
            inference_engine.config.merge_gap = float(config["merge_gap"])
        if "pitch_min" in config and config["pitch_min"] is not None:
            inference_engine.config.pitch_min = int(config["pitch_min"])
        if "pitch_max" in config and config["pitch_max"] is not None:
            inference_engine.config.pitch_max = int(config["pitch_max"])
        if "enable_fallback_segmentation" in config:
            inference_engine.config.enable_fallback_segmentation = bool(config["enable_fallback_segmentation"])
        # Instrument naming (optional)
        if "instrument_name" in config and config["instrument_name"]:
            inference_engine.config.instrument_name = str(config["instrument_name"])
        
        # Update progress
        job["progress"] = 0.3
        
        # Run transcription
        output_path = Path(API_CONFIG["output_dir"]) / f"{job_id}.mid"
        # Optional: quick preprocessing log for diagnostics
        try:
            mel = inference_engine._preprocess_audio(upload_path)
            logger.info(f" Mel shape: {mel.shape}")
        except Exception as e:
            logger.warning(f" Preprocessing log failed: {e}")
        
        start_time = datetime.now()
        midi = inference_engine.transcribe_audio(upload_path, output_path)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Update job completion
        job["status"] = "completed"
        job["message"] = "Transcription completed successfully"
        job["progress"] = 1.0
        job["midi_url"] = f"/download/{job_id}"
        job["completed_at"] = end_time.isoformat()
        job["processing_time"] = processing_time
        job["num_notes"] = len(midi.instruments[0].notes) if midi.instruments else 0
        
        logger.success(f"✅ Transcription completed: {job_id} ({job['num_notes']} notes)")
        
    except Exception as e:
        # Update job with error
        job["status"] = "failed"
        job["message"] = "Transcription failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now().isoformat()
        
        logger.error(f"❌ Transcription failed for job {job_id}: {str(e)}")


def run_server(host: str = "0.0.0.0", port: int = 8000, model_path: Optional[str] = None):
    """
    Run the API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        model_path: Path to model checkpoint
    """
    if model_path:
        os.environ["MIDI_MODEL_PATH"] = model_path
    
    logger.info(f"🚀 Starting MIDI Generator API on {host}:{port}")
    
    uvicorn.run(
        "src.api.api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MIDI Generator API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", help="Path to model checkpoint")
    
    args = parser.parse_args()
    run_server(args.host, args.port, args.model_path)
