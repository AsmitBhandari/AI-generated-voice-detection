from fastapi import FastAPI, HTTPException, Depends, Header, Request
from pydantic import BaseModel
from typing import Optional, Union
import uvicorn
import base64
import io
import torch
import numpy as np
import os
import requests
import tempfile
from sqlalchemy.orm import Session
from datetime import datetime

# Local imports
# Ensure we can import from ml folder
import sys
sys.path.append(".") 
from ml.preprocessing import load_audio, trim_silence, normalize_loudness
from ml.features import extractor
from ml.model import ModelWrapper
from app.database import SessionLocal, RequestLog

app = FastAPI(title="AI Voice Detector", version="1.0")

# --- Dependencies ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

API_KEY = os.getenv("API_KEY", "eternity")

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# --- Models ---
class VoiceRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str

class Explainability(BaseModel):
    spectral_entropy: float
    temporal_consistency: float

class VoiceResponse(BaseModel):
    prediction: str  # "ai_generated" or "human"
    confidence: float
    explainability: Explainability

class ErrorResponse(BaseModel):
    error: str

# --- Global Model Loader ---
# Load production model with versioning support
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.1.0")
MODEL_PATH = f"models/production/model_{MODEL_VERSION}"

# For hackathon integration, we prefer loading from HF if production path doesn't exist locally
# This ensures it works on Railway without manual folder uploads
local_model_exists = os.path.exists(MODEL_PATH)
if local_model_exists:
    print(f"Loading local production model: {MODEL_PATH}")
    model_wrapper = ModelWrapper(model_path=MODEL_PATH)
else:
    print(f"Production model {MODEL_PATH} not found locally. Loading from Hugging Face Hub...")
    model_wrapper = ModelWrapper() # Default loads from HF target established earlier
# --- Audio Download Helper ---
def download_audio_from_url(url: str, timeout: int = 10) -> bytes:
    """
    Download audio from URL.
    
    Args:
        url: Public URL to audio file
        timeout: Request timeout in seconds
    
    Returns:
        Audio bytes
    
    Raises:
        HTTPException: If download fails
    """
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'audio' not in content_type and 'octet-stream' not in content_type:
            # Still try to process, might be misconfigured server
            pass
        
        # Read content
        audio_bytes = response.content
        
        if len(audio_bytes) < 1000:
            raise HTTPException(status_code=400, detail="Downloaded audio is too small or empty")
        
        return audio_bytes
        
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Audio download timeout")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error downloading audio: {str(e)}")

# --- Explainability Helper ---
def compute_explainability(mel: torch.Tensor, phase: torch.Tensor):
    """
    Compute numeric metrics for explainability.
    """
    # 1. Spectral Entropy
    mel_np = mel.squeeze().cpu().numpy()
    psd = np.mean(mel_np, axis=1)
    psd = psd - np.min(psd)
    if np.sum(psd) > 0:
        p = psd / np.sum(psd)
        entropy = -np.sum(p * np.log(p + 1e-9))
    else:
        entropy = 0.0
    
    # 2. Temporal Consistency
    temp_var = np.var(np.mean(mel_np, axis=0))
    
    return {
        "spectral_entropy": float(entropy),
        "temporal_consistency": float(temp_var)
    }

def generate_explanation(prediction: str, spectral_entropy: float, temporal_consistency: float) -> dict:
    """
    Generate human-readable explanation based on audio features.
    
    Args:
        prediction: "ai_generated" or "human"
        spectral_entropy: Spectral entropy value (typically 0-5)
        temporal_consistency: Temporal variance value (typically 0-20)
    
    Returns:
        Dictionary with summary and detailed signal interpretations
    """
    # Thresholds for interpretation
    LOW_ENTROPY_THRESHOLD = 3.0
    HIGH_ENTROPY_THRESHOLD = 4.5
    LOW_VARIANCE_THRESHOLD = 3.0
    HIGH_VARIANCE_THRESHOLD = 8.0
    
    signals = {}
    
    # Spectral Entropy Interpretation
    if spectral_entropy < LOW_ENTROPY_THRESHOLD:
        entropy_interp = "Over-smooth spectral patterns typical of neural speech synthesis."
        entropy_indicator = "AI-like"
    elif spectral_entropy > HIGH_ENTROPY_THRESHOLD:
        entropy_interp = "Rich spectral complexity and natural variability found in human speech."
        entropy_indicator = "Human-like"
    else:
        entropy_interp = "Balanced spectral distribution seen in both human and synthetic speech."
        entropy_indicator = "Neutral"
    
    signals["spectral_entropy"] = {
        "value": round(spectral_entropy, 2),
        "interpretation": entropy_interp,
        "indicator": entropy_indicator
    }
    
    # Temporal Consistency Interpretation
    if temporal_consistency < LOW_VARIANCE_THRESHOLD:
        temporal_interp = "Highly consistent energy patterns suggest reduced natural jitter in AI speech."
        temporal_indicator = "AI-like"
    elif temporal_consistency > HIGH_VARIANCE_THRESHOLD:
        temporal_interp = "Natural fluctuations in speech energy with pauses and emphasis variations."
        temporal_indicator = "Human-like"
    else:
        temporal_interp = "Moderate temporal variation present in both human and advanced synthesis."
        temporal_indicator = "Neutral"
    
    signals["temporal_consistency"] = {
        "value": round(temporal_consistency, 2),
        "interpretation": temporal_interp,
        "indicator": temporal_indicator
    }
    
    # Generate Summary
    if prediction == "ai_generated":
        if entropy_indicator == "AI-like" and temporal_indicator == "AI-like":
            summary = "AI-generated: smooth spectral patterns and reduced temporal variability."
        elif entropy_indicator == "AI-like":
            summary = "AI-generated: over-smooth frequency distributions detected."
        elif temporal_indicator == "AI-like":
            summary = "AI-generated: highly consistent energy patterns detected."
        else:
            summary = "AI-generated: subtle synthetic patterns in acoustic features."
    
    else:  # human
        if entropy_indicator == "Human-like" and temporal_indicator == "Human-like":
            summary = "Human speech: natural spectral complexity and organic variations."
        elif entropy_indicator == "Human-like":
            summary = "Human speech: rich spectral complexity detected."
        elif temporal_indicator == "Human-like":
            summary = "Human speech: natural temporal fluctuations detected."
        else:
            summary = "Human speech: natural variability in acoustic features."
    
    return {
        "summary": summary,
        "signals": signals
    }

# --- Routes ---
@app.post("/predict")
@app.post("/detect-voice") # Keep as alias for backward compatibility
async def predict(
    req: VoiceRequest,
    request: Request,
    x_api_key: str = Header(...),
    db: Session = Depends(get_db)
):
    """
    Hackathon Test Endpoint compatible prediction.
    Accepts: language, audio_format, audio_base64
    """
    start_time = datetime.utcnow()
    
    # 0. Verify API key
    if x_api_key != API_KEY:
        return {"error": "Invalid API Key"}
    
    try:
        # 1. Decode Base64 Audio
        if not req.audio_base64:
            return {"error": "audio_base64 is required"}
            
        try:
            audio_bytes = base64.b64decode(req.audio_base64)
        except Exception:
            return {"error": "Invalid Base64 encoding"}
        
        # Validate audio size
        if len(audio_bytes) < 1000:
            return {"error": "Audio too short (less than 1KB)"}
        
        # 2. Load and Preprocess Audio
        try:
            waveform = load_audio(audio_bytes)
        except Exception as e:
            return {"error": f"Invalid or unsupported audio format: {str(e)}"}
        
        # Check duration
        duration = waveform.shape[1] / 16000.0
        if duration < 0.5:
            return {"error": "Audio duration too short (minimum 0.5s)"}
        
        # Trim silence and normalize
        try:
            waveform = trim_silence(waveform)
            waveform = normalize_loudness(waveform)
        except Exception as e:
            # Non-critical, continue with original waveform
            pass
        
        # Crop/pad to 4 seconds
        max_frames = 16000 * 4
        current_frames = waveform.shape[1]
        if current_frames > max_frames:
            start = (current_frames - max_frames) // 2
            waveform_in = waveform[:, start:start+max_frames]
        else:
            pad = max_frames - current_frames
            waveform_in = torch.nn.functional.pad(waveform, (0, pad))
        
        # 3. Extract Features (for explainability consistency)
        try:
            feats = extractor.extract_all(waveform_in)
        except Exception as e:
            return {"error": f"Feature extraction failed: {str(e)}"}
        
        # 4. Run Inference (Wav2Vec2)
        try:
            # Convert torch tensor (1, T) to 1D numpy array
            waveform_np = waveform_in.squeeze().numpy()
            confidence = model_wrapper.predict_from_waveform(waveform_np)
        except Exception as e:
            return {"error": f"Model inference failed: {str(e)}"}
        
        # 5. Determine Prediction
        # raw confidence is AI probability (0 to 1)
        if confidence > 0.5:
            prediction = "ai_generated"
            display_confidence = confidence
        else:
            prediction = "human"
            display_confidence = 1.0 - confidence
        
        # 6. Compute Explainability
        try:
            metrics = compute_explainability(feats['mel'], feats['phase'])
            
            # Generate human-readable explanation
            explanation = generate_explanation(
                prediction=prediction,
                spectral_entropy=metrics["spectral_entropy"],
                temporal_consistency=metrics["temporal_consistency"]
            )
        except Exception as e:
            # Fallback explanation
            explanation = {
                "summary": "Unable to generate detailed explanation due to processing error.",
                "signals": {
                    "spectral_entropy": {
                        "value": 0.0,
                        "interpretation": "Feature extraction failed.",
                        "indicator": "Unknown"
                    },
                    "temporal_consistency": {
                        "value": 0.0,
                        "interpretation": "Feature extraction failed.",
                        "indicator": "Unknown"
                    }
                }
            }
        
        # 7. Log to Database (non-blocking)
        try:
            log_entry = RequestLog(
                api_key=x_api_key,
                language=req.language,
                duration_sec=duration,
                result=prediction,
                confidence=confidence,
                client_ip=request.client.host if request.client else "unknown"
            )
            db.add(log_entry)
            db.commit()
        except Exception as e:
            # Don't fail request if logging fails
            print(f"Warning: Failed to log request: {e}")
        
        # 8. Return Response
        return {
            "prediction": prediction,
            "confidence": float(display_confidence),
            "explanation": explanation
        }
    
    except Exception as e:
        # Catch-all for any unexpected errors
        print(f"Unexpected error in detect_voice: {e}")
        return {"error": f"Internal server error: {str(e)}"}

@app.get("/")
def home():
    return {
        "message": "AI Voice Detector API is running",
        "version": "1.0",
        "endpoints": {
            "detect": "POST /detect-voice"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "model_version": model_wrapper.model_version,
        "model_loaded": model_wrapper.model is not None
    }
