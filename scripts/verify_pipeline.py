import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

OUTPUT_ROOT = Path("output")

def verify_pipeline():
    print("="*60)
    print("PIPELINE VERIFICATION")
    print("="*60)
    
    if not OUTPUT_ROOT.exists():
        print(f"Output root {OUTPUT_ROOT} does not exist!")
        return

    # Counts: splits -> class -> lang -> count
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # Walk
    for split in ["train", "val", "test"]:
        split_dir = OUTPUT_ROOT / split
        if not split_dir.exists(): continue
        
        for cls in ["human", "ai"]:
            cls_dir = split_dir / cls
            if not cls_dir.exists(): continue
            
            for f in cls_dir.rglob("*.wav"):
                # Path: split/class/lang/...
                rel = f.relative_to(cls_dir)
                lang = rel.parts[0]
                counts[split][cls][lang] += 1

    # Display
    data = []
    
    langs = set()
    for s in counts:
        for c in counts[s]:
            langs.update(counts[s][c].keys())
            
    sorted_langs = sorted(list(langs))
    
    print(f"{'Split':<10} | {'Class':<10} | {'Total':<8} | " + " | ".join([f"{l:<10}" for l in sorted_langs]))
    print("-" * (36 + 13 * len(sorted_langs)))
    
    total_samples = 0
    
    for split in ["train", "val", "test"]:
        for cls in ["human", "ai"]:
            row_counts = [counts[split][cls][l] for l in sorted_langs]
            row_total = sum(row_counts)
            total_samples += row_total
            
            row_str = f"{split:<10} | {cls:<10} | {row_total:<8} | " + " | ".join([f"{c:<10}" for c in row_counts])
            print(row_str)
            
    print("-" * (36 + 13 * len(sorted_langs)))
    print(f"Grand Total Samples: {total_samples}")
    
    # Check Ratios
    train_n = sum(sum(counts["train"][c].values()) for c in counts["train"])
    val_n = sum(sum(counts["val"][c].values()) for c in counts["val"])
    test_n = sum(sum(counts["test"][c].values()) for c in counts["test"])
    
    print("\nSplit Ratios:")
    if total_samples > 0:
        print(f"Train: {train_n} ({train_n/total_samples:.1%}) (Target: 80%)")
        print(f"Val:   {val_n}   ({val_n/total_samples:.1%})   (Target: 10%)")
        print(f"Test:  {test_n}  ({test_n/total_samples:.1%})  (Target: 10%)")
    
if __name__ == "__main__":
    verify_pipeline()
