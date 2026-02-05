import os
import torch
import soundfile as sf
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path

# Target languages (English subsets from minds14)
langs = ["english_us", "english_au", "english_gb"]
# Mapping to MMS-TTS model codes
lang_code_map = {
    "english_us": "eng",
    "english_au": "eng",
    "english_gb": "eng"
}

data_dir = Path("data/real")
output_dir = Path("data/fake")
output_dir.mkdir(exist_ok=True)

device = 0 if torch.cuda.is_available() else -1

print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

for lang in langs:
    print(f"\nSynthesizing AI voices for {lang}...")
    model_id = f"facebook/mms-tts-{lang_code_map[lang]}"
    
    try:
        # Load TTS pipeline
        tts = pipeline("text-to-speech", model=model_id, device=device)
        
        # Find all transcripts for this language
        files = list(data_dir.glob(f"{lang}_*.txt"))
        
        for txt_file in tqdm(files, desc=f"Synthesizing {lang}"):
            idx = txt_file.stem.split('_')[-1]
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
                
            if not text:
                continue
                
            # Synthesize
            output = tts(text)
            
            # Save audio
            output_path = output_dir / f"{lang}_{idx}.wav"
            sf.write(str(output_path), output["audio"], output["sampling_rate"])
            
    except Exception as e:
        print(f"Error synthesizing {lang}: {e}")

print(f"\n✅ Successfully synthesized AI samples to {output_dir}")
