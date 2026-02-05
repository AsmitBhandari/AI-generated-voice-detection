import os
import shutil
import random
from pathlib import Path

OUTPUT_ROOT = Path("output")
LANGS = ["hindi", "tamil", "telugu", "malayalam"]

def rebalance_indic():
    print("Rebalancing Indic Human data for Verification...")
    
    for lang in LANGS:
        # Source: output/train/human/{lang}
        # Dest: output/test/human/{lang}
        
        src_dir = OUTPUT_ROOT / "train" / "human" / lang
        dest_dir = OUTPUT_ROOT / "test" / "human" / lang
        
        if not src_dir.exists():
            print(f"Source not found: {src_dir}")
            continue
            
        files = list(src_dir.rglob("*.wav"))
        print(f"  {lang}: Found {len(files)} training samples.")
        
        target_test = 15
        
        if len(files) <= target_test:
            print(f"    Not enough files to move.")
            continue
            
        # Select random
        to_move = random.sample(files, target_test)
        
        os.makedirs(dest_dir, exist_ok=True)
        
        for f in to_move:
            dest_path = dest_dir / f.name
            shutil.move(str(f), str(dest_path))
            
        print(f"    Moved {len(to_move)} files to {dest_dir}")
        
    print("Rebalance complete.")

if __name__ == "__main__":
    rebalance_indic()
