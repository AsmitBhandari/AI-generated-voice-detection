import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict


def compute_dataset_fingerprint(dataset_root: str) -> Dict[str, Any]:
    """
    Generate comprehensive dataset fingerprint.
    
    Creates a manifest of all audio files and computes:
    - SHA256 hash of the manifest
    - Sample counts per split
    - Class balance
    - Language distribution
    - Speaker count
    
    Args:
        dataset_root: Root directory of processed dataset
                     Expected structure: dataset_root/{split}/{class}/{language}/{speaker}/{file.wav}
    
    Returns:
        Dictionary with dataset metadata
    """
    dataset_path = Path(dataset_root)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    manifest = []
    split_counts = defaultdict(int)
    class_balance = {"human": 0, "ai": 0}
    languages = set()
    speakers = set()
    
    print(f"Computing dataset fingerprint for: {dataset_root}")
    
    # Scan all splits
    for split in ["train", "val", "test"]:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"  Warning: Split '{split}' not found, skipping...")
            continue
        
        # Scan all classes
        for cls in ["human", "ai"]:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            
            # Find all .wav files recursively
            for audio_file in cls_dir.rglob("*.wav"):
                # Extract metadata from path structure
                # Relative path from split_dir: class/language/speaker/file.wav
                rel_parts = audio_file.relative_to(split_dir).parts
                
                lang = "unknown"
                speaker = "unknown"
                
                # Parse structure
                if len(rel_parts) >= 2:
                    # At least: class/language/...
                    lang = rel_parts[1]
                    languages.add(lang)
                
                if len(rel_parts) >= 3:
                    # class/language/speaker/...
                    speaker = rel_parts[2]
                    speakers.add(f"{lang}_{speaker}")  # Unique speaker ID
                
                # Add to manifest
                manifest.append({
                    "path": str(audio_file.relative_to(dataset_path)),
                    "split": split,
                    "class": cls,
                    "language": lang,
                    "speaker": speaker,
                    "size": audio_file.stat().st_size
                })
                
                split_counts[split] += 1
                
                # Count class balance for training set only
                if split == "train":
                    class_balance[cls] += 1
    
    # Sort manifest for deterministic hashing
    manifest.sort(key=lambda x: x["path"])
    
    # Compute SHA256 hash
    manifest_str = json.dumps(manifest, sort_keys=True)
    dataset_hash = hashlib.sha256(manifest_str.encode()).hexdigest()
    
    metadata = {
        "dataset_hash": dataset_hash,
        "num_train_samples": split_counts.get("train", 0),
        "num_val_samples": split_counts.get("val", 0),
        "num_test_samples": split_counts.get("test", 0),
        "class_balance": class_balance,
        "languages": sorted(list(languages)),
        "speaker_count": len(speakers),
        "total_files": len(manifest)
    }
    
    print(f"✓ Dataset fingerprint computed:")
    print(f"  Hash: {dataset_hash[:16]}...")
    print(f"  Train: {metadata['num_train_samples']}, Val: {metadata['num_val_samples']}, Test: {metadata['num_test_samples']}")
    print(f"  Class balance: {class_balance}")
    print(f"  Languages: {metadata['languages']}")
    print(f"  Speakers: {metadata['speaker_count']}")
    
    return metadata


def save_manifest(dataset_root: str, output_path: str):
    """
    Save dataset manifest to JSON file for inspection.
    
    Args:
        dataset_root: Root directory of dataset
        output_path: Path to save manifest JSON
    """
    metadata = compute_dataset_fingerprint(dataset_root)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Manifest saved: {output_path}")


def verify_dataset_integrity(dataset_root: str, expected_hash: str) -> bool:
    """
    Verify dataset hasn't changed by comparing hash.
    
    Args:
        dataset_root: Root directory of dataset
        expected_hash: Expected SHA256 hash
    
    Returns:
        True if hash matches, False otherwise
    """
    metadata = compute_dataset_fingerprint(dataset_root)
    current_hash = metadata["dataset_hash"]
    
    if current_hash == expected_hash:
        print(f"✓ Dataset integrity verified: {current_hash[:16]}...")
        return True
    else:
        print(f"✗ Dataset hash mismatch!")
        print(f"  Expected: {expected_hash[:16]}...")
        print(f"  Current:  {current_hash[:16]}...")
        return False
