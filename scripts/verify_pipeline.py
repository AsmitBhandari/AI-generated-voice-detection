import torch
import numpy as np
import scipy.io.wavfile as wav
import io
import sys
import os

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocessing import load_audio, trim_silence, pad_or_crop, augment_noise, extractor
from ml.model import DualBranchModel

def test_pipeline():
    print("--- Starting Pipeline Verification ---")
    
    # 1. Create Dummy Audio (Sine sweep)
    sr = 16000
    t = np.linspace(0, 3, sr * 3)
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Save to bytes to simulate file load
    byte_io = io.BytesIO()
    wav.write(byte_io, sr, audio_data)
    byte_io.seek(0)
    
    print("[1] Audio Created")
    
    # 2. Preprocessing
    waveform = load_audio(byte_io)
    print(f"[2] Loaded Waveform: {waveform.shape}")
    
    waveform = trim_silence(waveform)
    print(f"[2] Trimmed: {waveform.shape}")
    
    waveform = augment_noise(waveform)
    print(f"[2] Augmented (Noise): {waveform.shape}")
    
    # Pad to fixed size
    waveform = pad_or_crop(waveform, max_len_sec=4.0)
    print(f"[2] Padded/Cropped: {waveform.shape}")
    
    # 3. Features
    feats = extractor.extract_all(waveform)
    print(f"[3] Mel Shape: {feats['mel'].shape}")     # Expect (1, 80, T)
    print(f"[3] Phase Shape: {feats['phase'].shape}") # Expect (1, 80, T)
    
    # 4. Model
    model = DualBranchModel() # Init weights
    print("[4] Model Initialized")
    
    # Forward Pass
    # Add batch dim
    mel = feats['mel'].unsqueeze(0)
    phase = feats['phase'].unsqueeze(0)
    
    with torch.no_grad():
        logits = model(mel, phase)
        prob = torch.sigmoid(logits).item()
        
    print(f"[5] Prediction: Logits={logits.item():.4f}, Prob={prob:.4f}")
    print("--- Verification Success ---")

if __name__ == "__main__":
    test_pipeline()
