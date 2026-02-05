import os
from datasets import load_dataset, Audio
from tqdm import tqdm
import torch
import torchaudio
from pathlib import Path

# Target languages and their FLEURS config names
# hi_in (Hindi), ta_in (Tamil), te_in (Telugu), ml_in (Malayalam)
langs = {
    "hindi": "hi_in",
    "tamil": "ta_in",
    "telugu": "te_in",
    "malayalam": "ml_in"
}

# Output root - following existing convention
RAW_ROOT = Path("dataset/raw/human")

def download_indic_human():
    """
    Downloads human speech from Google FLEURS.
    Target: 150 samples per language.
    """
    for lang_name, config_name in langs.items():
        print(f"\nProcessing {lang_name} ({config_name})...")
        
        try:
            # Load dataset in STREAMING mode (much faster)
            ds = load_dataset(
                "google/fleurs", 
                config_name, 
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            
            lang_dir = RAW_ROOT / lang_name
            os.makedirs(lang_dir, exist_ok=True)
            
            count = 0
            limit = 150
            
            print(f"Downloading {lang_name} (target: {limit} samples)...")
            
            for idx, sample in enumerate(ds):
                if count >= limit:
                    break
                
                audio = sample['audio']
                array = audio['array']
                sr = audio['sampling_rate']
                duration = len(array) / sr
                
                # Filter by duration (2s to 10s)
                if not (2.0 <= duration <= 10.0):
                    continue
                
                # Extract speaker info if available, otherwise use index
                # FLEURS has 'speaker_id'
                speaker_id = str(sample.get('speaker_id', 'unknown'))
                speaker_dir = lang_dir / speaker_id
                os.makedirs(speaker_dir, exist_ok=True)
                
                # Save audio as wav
                file_id = f"fleurs_{count}"
                save_path = speaker_dir / f"{file_id}.wav"
                
                # Save matching transcript for synthesis
                text_path = speaker_dir / f"{file_id}.txt"
                
                if not save_path.exists():
                    # Save at 16kHz mono 
                    # If sr is not 16000, we should resample, but FLEURS is typically 16k or higher
                    # We'll let the existing pipeline handle final normalization, 
                    # but we save it here in a standard format.
                    audio_tensor = torch.from_numpy(array).unsqueeze(0).float()
                    if sr != 16000:
                        resampler = torchaudio.transforms.Resample(sr, 16000)
                        audio_tensor = resampler(audio_tensor)
                        sr = 16000
                    
                    # Save with soundfile to avoid torchaudio backend issues
                    import soundfile as sf
                    sf.write(str(save_path), audio_tensor.squeeze().numpy(), sr)
                    
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(sample['raw_transcription'])
                        
                    count += 1
                    if count % 25 == 0:
                        print(f"  Downloaded {count}/{limit} samples...")
            
            print(f"[OK] Downloaded {count} {lang_name} samples")
            
        except Exception as e:
            print(f"Error downloading {lang_name}: {e}")

if __name__ == "__main__":
    download_indic_human()
