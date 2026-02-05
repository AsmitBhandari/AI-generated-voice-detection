import torch
import torchaudio
import librosa
import numpy as np
import io
import random
from typing import Optional, Union

SAMPLE_RATE = 16000
MAX_DURATION = 6.0 # seconds
MIN_DURATION = 1.0 # seconds

def load_audio(source: Union[str, bytes, io.BytesIO], target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Load audio from path or bytes, convert to mono, and resample.
    Returns: Tensor of shape (1, T)
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    
    try:
        waveform, sr = torchaudio.load(source)
    except Exception as e:
        # Fallback for some formats or raw bytes issues if torchaudio fails
        import soundfile as sf
        if isinstance(source, io.BytesIO):
             source.seek(0)
        data, sr = sf.read(source)
        waveform = torch.from_numpy(data.T).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        
    return waveform

def trim_silence(waveform: torch.Tensor, threshold_db: float = 20.0) -> torch.Tensor:
    """
    Trim silence from beginning and end using librosa.
    """
    # Convert to numpy for librosa
    np_wav = waveform.squeeze().numpy()
    non_silent_intervals = librosa.effects.split(np_wav, top_db=threshold_db)
    
    if len(non_silent_intervals) == 0:
        return waveform # Return original if all silent (or handle as error later)
        
    # Concatenate all non-silent parts? Or just start/end? 
    # Usually we want to keep internal structure but remove lead/trail.
    # librosa.effects.trim does lead/trail.
    trimmed, _ = librosa.effects.trim(np_wav, top_db=threshold_db)
    
    return torch.from_numpy(trimmed).unsqueeze(0)

def normalize_loudness(waveform: torch.Tensor, target_db: float = -20.0) -> torch.Tensor:
    """
    Normalize audio loudness. Simple peak or RMS normalization.
    """
    # Using RMS normalization here
    rms = torch.sqrt(torch.mean(waveform ** 2))
    if rms < 1e-6:
        return waveform
        
    scalar = (10 ** (target_db / 20)) / (rms + 1e-9)
    return waveform * scalar

def pad_or_crop(waveform: torch.Tensor, max_len_sec: float = MAX_DURATION) -> torch.Tensor:
    num_frames = int(max_len_sec * SAMPLE_RATE)
    current_frames = waveform.shape[1]
    
    if current_frames > num_frames:
        # Random crop for training, center for inference? 
        # For this util, let's do center crop default, caller can do random.
        start = (current_frames - num_frames) // 2
        return waveform[:, start:start+num_frames]
    elif current_frames < num_frames:
        padding = num_frames - current_frames
        return torch.nn.functional.pad(waveform, (0, padding))
    
    return waveform

# --- Robustness Augmentations ---

def augment_mp3_reencode(waveform: torch.Tensor, bitrate: str = "64k") -> torch.Tensor:
    """
    Simulate MP3 compression artifacts by saving to memory buffer and reloading.
    Requires ffmpeg available in system.
    """
    try:
        # Save to buffer
        src_buf = io.BytesIO()
        torchaudio.save(src_buf, waveform, SAMPLE_RATE, format="mp3", compression=bitrate) # compression arg varies by backed
        # Note: torchaudio.save with mp3 might fail if underlying lib unavailable.
        # Alternative: use pydub if installed, but let's try torchaudio first or fallback.
        
        src_buf.seek(0)
        reloaded, _ = torchaudio.load(src_buf, format="mp3")
        return reloaded
    except Exception as e:
        # print(f"MP3 augment failed: {e}")
        return waveform # Fail gracefully

def augment_noise(waveform: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
    """
    Add Gaussian noise.
    """
    noise = torch.randn_like(waveform)
    
    # Calculate signal power
    s_pow = torch.mean(waveform ** 2)
    n_pow = torch.mean(noise ** 2)
    
    # Calculate scale needed for noise
    if s_pow == 0:
        return waveform
        
    target_n_pow = s_pow / (10 ** (snr_db / 10))
    scale = torch.sqrt(target_n_pow / (n_pow + 1e-9))
    
    return waveform + noise * scale
