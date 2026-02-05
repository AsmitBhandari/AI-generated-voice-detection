import os
from datasets import load_dataset, Audio
from tqdm import tqdm
import torch
import librosa
from pathlib import Path

# Target languages and their Common Voice config names
langs = {
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
    "malayalam": "ml"
}

# Output root
RAW_ROOT = Path("dataset/raw/human")

def download_indic_human():
    """
    Downloads validated human speech from Mozilla Common Voice.
    Target: 150 samples per language.
    """
    for lang_name, lang_code in langs.items():
        print(f"\nProcessing {lang_name} ({lang_code})...")
        
        try:
            # Load dataset (streaming for speed/efficiency)
            ds = load_dataset(
                "mozilla-foundation/common_voice_11_0", 
                lang_code, 
                split="train", 
                streaming=True,
                use_auth_token=True # Requires HF login
            )
            
            lang_dir = RAW_ROOT / lang_name
            os.makedirs(lang_dir, exist_ok=True)
            
            count = 0
            limit = 150
            
            # Use iterator to get validated samples
            pbar = tqdm(total=limit, desc=f"Downloading {lang_name}")
            
            for sample in ds:
                if count >= limit:
                    break
                
                # Check validation (Common Voice 11 has 'up_votes')
                if sample.get('up_votes', 0) < 1:
                    continue
                
                # Check duration (approximate by path or better yet, load)
                # For streaming, we need to load the audio to check duration
                audio = sample['audio']
                array = audio['array']
                sr = audio['sampling_rate']
                duration = len(array) / sr
                
                if not (2.0 <= duration <= 10.0):
                    continue
                
                # Structure: dataset/raw/human/<lang>/<speaker>/<id>.wav
                client_id = sample.get('client_id', 'unknown_speaker')[:12] # Shorten ID
                speaker_dir = lang_dir / client_id
                os.makedirs(speaker_dir, exist_ok=True)
                
                # Save audio
                file_id = Path(sample['path']).stem
                save_path = speaker_dir / f"{file_id}.wav"
                
                # Save matching transcript for synthesis
                text_path = speaker_dir / f"{file_id}.txt"
                
                if not save_path.exists():
                    # Convert to 16kHz mono (standard for pipeline)
                    librosa.output.write_wav = lambda path, data, samplerate: torchaudio.save(path, torch.from_numpy(data).unsqueeze(0), samplerate) # Shim
                    import torchaudio
                    torchaudio.save(str(save_path), torch.from_numpy(array).unsqueeze(0), sr)
                    
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(sample['sentence'])
                        
                    count += 1
                    pbar.update(1)
            
            pbar.close()
            
        except Exception as e:
            print(f"Error downloading {lang_name}: {e}")
            print("TIP: Ensure you have accepted the dataset terms at https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0")

if __name__ == "__main__":
    download_indic_human()
