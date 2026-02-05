import os
import torch
import soundfile as sf
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path

# Target languages and their MMS-TTS model codes
lang_config = {
    "hindi": "hin",
    "tamil": "tam",
    "telugu": "tel",
    "malayalam": "mal"
}

HUMAN_ROOT = Path("dataset/raw/human")
AI_ROOT = Path("dataset/raw/ai")

device = 0 if torch.cuda.is_available() else -1

def synthesize_indic_ai():
    """
    Scans dataset/raw/human for transcripts and generates matching AI speech.
    Saves to dataset/raw/ai/<lang>/mms_tts/<file>.wav
    """
    for lang_name, mms_code in lang_config.items():
        print(f"\nSynthesizing AI voices for {lang_name} ({mms_code})...")
        
        lang_human_dir = HUMAN_ROOT / lang_name
        lang_ai_dir = AI_ROOT / lang_name / "mms_tts"
        os.makedirs(lang_ai_dir, exist_ok=True)
        
        if not lang_human_dir.exists():
            print(f"Skipping {lang_name}: Human data directory not found.")
            continue
            
        model_id = f"facebook/mms-tts-{mms_code}"
        
        try:
            # Load TTS pipeline
            tts = pipeline("text-to-speech", model=model_id, device=device)
            
            # Find all transcripts recursively in the human folder
            files = list(lang_human_dir.glob("**/*.txt"))
            
            if not files:
                print(f"No transcripts found for {lang_name}.")
                continue
                
            for txt_file in tqdm(files, desc=f"Synthesizing {lang_name}"):
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    
                if not text:
                    continue
                
                # Use same filename for the wav
                file_id = txt_file.stem
                output_path = lang_ai_dir / f"{file_id}.wav"
                
                if output_path.exists():
                    continue
                    
                # Synthesize
                output = tts(text)
                
                # Save audio
                sf.write(str(output_path), output["audio"], output["sampling_rate"])
                
        except Exception as e:
            print(f"Error synthesizing {lang_name}: {e}")

if __name__ == "__main__":
    synthesize_indic_ai()
