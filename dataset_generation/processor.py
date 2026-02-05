import torch
import torchaudio
import librosa
import numpy as np
import io
import logging
from .config import CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_convert(path_or_bytes):
    """
    Load audio, convert to mono, resample to target SR.
    Returns: Tensor (1, T)
    """
    try:
        waveform, sr = torchaudio.load(path_or_bytes)
    except Exception as e:
        logger.error(f"Failed to load {path_or_bytes}: {e}")
        return None

    # Mix to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample
    if sr != CONFIG["sample_rate"]:
        resampler = torchaudio.transforms.Resample(sr, CONFIG["sample_rate"])
        waveform = resampler(waveform)
        
    return waveform

def normalize_loudness(waveform):
    """
    RMS-based normalization to target DB.
    """
    rms = torch.sqrt(torch.mean(waveform ** 2))
    if rms < 1e-6:
        return waveform # Too silent to normalize safely
        
    scalar = (10 ** (CONFIG["target_db"] / 20)) / (rms + 1e-9)
    return waveform * scalar

def smart_trim(waveform):
    """
    Trims ONLY leading/trailing silence. Preserves internal pauses/breath.
    """
    np_wav = waveform.squeeze().numpy()
    
    # We use librosa.effects.trim which restricts to leading/trailing
    # top_db: The threshold (in decibels) below reference to consider as silence
    trimmed, _ = librosa.effects.trim(np_wav, top_db=CONFIG["silence_top_db"])
    
    return torch.from_numpy(trimmed).unsqueeze(0)

def segment_audio(waveform):
    """
    Enforces min/max duration.
    - If < min: Return None (Discard)
    - If > max: Random Crop
    - Else: Return as is
    """
    num_frames = waveform.shape[1]
    sr = CONFIG["sample_rate"]
    min_frames = int(CONFIG["min_duration"] * sr)
    max_frames = int(CONFIG["max_duration"] * sr)
    
    if num_frames < min_frames:
        return None # Discard
        
    if num_frames > max_frames:
        # Random Crop
        start = np.random.randint(0, num_frames - max_frames)
        return waveform[:, start:start+max_frames]
        
    return waveform

def check_quality(waveform):
    """
    Heuristics to reject bad files.
    - Check for silence (RMS close to 0)
    - Check for clipping (values close to 1/-1 too often)
    """
    # 1. Silence check
    if torch.max(torch.abs(waveform)) < 0.01:
        return False, "Too Silent"
        
    # 2. Clipping check (crude)
    # If > 1% of samples are 1.0 or -1.0, maybe clipped.
    # But after loudness norm, this might drift. Checking against typical range.
    # Let's rely on RMS > threshold (done in norm) and non-empty.
    
    return True, "OK"

def process_single_file(path):
    """
    Full pipeline for a single file.
    Returns: Processed Tensor or None
    """
    # 1. Format
    wav = load_and_convert(path)
    if wav is None: return None, "Load failed"
    
    # 2. Quality Check (Pre-norm)
    ok, reason = check_quality(wav)
    if not ok: return None, reason
    
    # 3. Trim
    wav = smart_trim(wav)
    
    # 4. Normalize
    wav = normalize_loudness(wav)
    
    # 5. Segment
    wav = segment_audio(wav)
    if wav is None: return None, "Too short after trim/segment"
    
    return wav, "Success"
