import os
import glob
import torchaudio
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch

def verify_dataset(data_root):
    print(f"Scanning {data_root}...")
    files = glob.glob(os.path.join(data_root, '**', '*.wav'), recursive=True)
    print(f"Found {len(files)} files.")
    
    durations = []
    loudness = []
    
    for f in files:
        try:
            wav, sr = torchaudio.load(f)
            # Check SR
            if sr != 16000:
                print(f"WARNING: Bad SR {sr} in {f}")
            
            # Duration
            dur = wav.shape[1] / sr
            durations.append(dur)
            
            # Loudness (RMS)
            rms = torch.sqrt(torch.mean(wav ** 2)).item()
            loudness.append(20 * np.log10(rms + 1e-9))
            
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not durations:
        print("No valid audio files found.")
        return

    # Plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(durations, bins=20, alpha=0.7)
    plt.title("Duration Distribution")
    plt.xlabel("Seconds")
    
    plt.subplot(1, 2, 2)
    plt.hist(loudness, bins=20, alpha=0.7, color='orange')
    plt.title("Loudness Distribution (dB)")
    plt.xlabel("dB")
    
    out_img = "dataset_stats.png"
    plt.savefig(out_img)
    print(f"Stats plot saved to {out_img}")
    
    print(f"Mean Duration: {np.mean(durations):.2f}s")
    print(f"Min Duration: {np.min(durations):.2f}s")
    print(f"Mean Loudness: {np.mean(loudness):.2f}dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    verify_dataset(args.data)
