import os
import shutil
from pathlib import Path
from tqdm import tqdm

AI_ROOT = Path("dataset/raw/ai")

def fix_ai_structure():
    """
    Moves files from dataset/raw/ai/{lang}/mms_tts/ to dataset/raw/ai/{lang}/
    and removes mms_tts directory.
    This ensures the pipeline treats each file as a separate group (random split)
    instead of treating 'mms_tts' as a single speaker (group).
    """
    print("=" * 60)
    print("FIXING AI DATA STRUCTURE")
    print("=" * 60)
    
    if not AI_ROOT.exists():
        print(f"Directory not found: {AI_ROOT}")
        return

    # Iterate over languages
    for lang_dir in AI_ROOT.iterdir():
        if not lang_dir.is_dir():
            continue
            
        mms_dir = lang_dir / "mms_tts"
        if mms_dir.exists() and mms_dir.is_dir():
            print(f"Processing {lang_dir.name}...")
            
            files = list(mms_dir.glob("*.wav"))
            if not files:
                print(f"  No files in {mms_dir}")
                continue
                
            for f in files:
                dest = lang_dir / f.name
                shutil.move(str(f), str(dest))
                
            # Remove empty directory
            try:
                mms_dir.rmdir()
                print(f"  Moved {len(files)} files and removed 'mms_tts' folder.")
            except OSError:
                print(f"  Moved {len(files)} files but could not remove 'mms_tts' (not empty?).")
        else:
            print(f"Skipping {lang_dir.name} (no mms_tts folder)")
            
    print("\n" + "=" * 60)
    print("STRUCTURE FIX COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    fix_ai_structure()
