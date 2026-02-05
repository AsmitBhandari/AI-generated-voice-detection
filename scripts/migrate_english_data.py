import os
import shutil
from pathlib import Path

# Source directories (existing English data)
SOURCE_HUMAN = Path("data/real")
SOURCE_AI = Path("data/fake")

# Destination directories (dataset/raw structure)
DEST_HUMAN = Path("dataset/raw/human/english")
DEST_AI = Path("dataset/raw/ai/english/mms_tts")

def migrate_english_data():
    """
    Migrate English data from data/real and data/fake to dataset/raw structure.
    This ensures all data (English + Indic) goes through the same preprocessing pipeline.
    """
    print("=" * 60)
    print("MIGRATING ENGLISH DATA TO DATASET/RAW STRUCTURE")
    print("=" * 60)
    
    # Create destination directories
    DEST_HUMAN.mkdir(parents=True, exist_ok=True)
    DEST_AI.mkdir(parents=True, exist_ok=True)
    
    # Migrate human (real) data
    print(f"\nMigrating human speech from {SOURCE_HUMAN}...")
    human_files = list(SOURCE_HUMAN.glob("*.wav"))
    
    if not human_files:
        print(f"[WARNING] No .wav files found in {SOURCE_HUMAN}")
    else:
        count = 0
        for src_file in human_files:
            dest_file = DEST_HUMAN / src_file.name
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)
                count += 1
                if count % 100 == 0:
                    print(f"  Copied {count}/{len(human_files)} files...")
        print(f"[OK] Migrated {len(human_files)} human samples")
    
    # Migrate AI (fake) data
    print(f"\nMigrating AI speech from {SOURCE_AI}...")
    ai_files = list(SOURCE_AI.glob("*.wav"))
    
    if not ai_files:
        print(f"[WARNING] No .wav files found in {SOURCE_AI}")
    else:
        count = 0
        for src_file in ai_files:
            dest_file = DEST_AI / src_file.name
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)
                count += 1
                if count % 100 == 0:
                    print(f"  Copied {count}/{len(ai_files)} files...")
        print(f"[OK] Migrated {len(ai_files)} AI samples")
    
    # Summary
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"Human data: {DEST_HUMAN}")
    print(f"AI data:    {DEST_AI}")
    print(f"\nTotal migrated: {len(human_files)} human + {len(ai_files)} AI = {len(human_files) + len(ai_files)} samples")
    print("\n[OK] English data is now ready for preprocessing pipeline")
    print("=" * 60)

if __name__ == "__main__":
    migrate_english_data()
