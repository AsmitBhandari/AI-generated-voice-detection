import os
from datasets import load_dataset, Audio
import soundfile as sf
from tqdm import tqdm

langs = ["hindi", "tamil", "telugu", "malayalam"]
target_dir = "data/real"
os.makedirs(target_dir, exist_ok=True)

samples_per_lang = 200

# Get token from environment
hf_token = os.environ.get("HF_TOKEN")

# MINDS14 language codes: en-US, en-AU, en-GB are guaranteed to work
lang_map = {
    "english_us": "en-US",
    "english_au": "en-AU",
    "english_gb": "en-GB",
}

for lang_name, lang_code in lang_map.items():
    print(f"Streaming {lang_name} ({lang_code}) samples from MINDS14...")
    try:
        # Load one language at a time
        # MINDS14 is available in parquet
        ds = load_dataset("PolyAI/minds14", lang_code, split="train", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False)) 
        
        count = 0
        pbar = tqdm(total=samples_per_lang, desc=f"Downloading {lang_name}")
        
        for sample in ds:
            if count >= samples_per_lang:
                break
            
            # sample["audio"] will contain {"path": ..., "bytes": ...}
            audio_bytes = sample["audio"]["bytes"]
            
            # Save audio array to wav
            file_name = f"{lang_name}_{count}.wav"
            file_path = os.path.join(target_dir, file_name)
            
            # Write bytes directly to file
            with open(file_path, "wb") as f:
                f.write(audio_bytes)
            
            # Save corresponding transcript for synthesis later
            transcript_path = os.path.join(target_dir, f"{lang_name}_{count}.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(sample["transcription"])
            
            count += 1
            pbar.update(1)
        pbar.close()
        
    except Exception as e:
        print(f"Error downloading {lang_name}: {e}")

print(f"\nSuccessfully downloaded human samples and transcripts to {target_dir}")
