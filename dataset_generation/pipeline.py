import os
import glob
import torch
import torchaudio
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from .config import CONFIG
from .processor import process_single_file
import logging

# Setup Logging
logging.basicConfig(filename='dataset_generation_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def get_files(src_dir, extensions=('*.mp3', '*.wav', '*.flac')):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(src_dir, '**', ext), recursive=True))
    return files

def run_pipeline(raw_root, output_root):
    """
    Robust pipeline processing.
    Assumes structure: raw_root/{class}/{language}/{speaker}/{file}
    OR: raw_root/{class}/{language}/{file}
    
    1. Identifies Language to ensure balanced splits.
    2. Identifies Speaker (if present) to prevent data leakage (Train/Test set contamination).
    """
    
    random.seed(CONFIG["random_seed"])
    
    # Check classes
    classes = ['human', 'ai']
    
    stats = {"processed": 0, "discarded": 0, "errors": 0}
    
    # Data Structure: map[class][language][group_id] -> list(files)
    # group_id is 'speaker' folder if exists, else filename
    dataset_map = {c: {} for c in classes}
    
    logging.info(f"Scanning {raw_root}...")
    
    all_files = get_files(raw_root)
    
    for fpath in all_files:
        rel_path = os.path.relpath(fpath, raw_root)
        parts = rel_path.split(os.sep)
        
        # parts[0] is class (human/ai)
        # parts[1] is language (en/hi/...) OR file if no lang
        
        if len(parts) < 2:
            continue # Skip files at root
            
        cls = parts[0]
        if cls not in classes:
            continue # Skip non-class folders
            
        # Determine Language and Grouping
        if len(parts) >= 3:
            # Structure: class/language/...
            lang = parts[1]
            
            if len(parts) >= 4:
                # Structure: class/language/speaker/file.wav
                # Group by Speaker (parts[2])
                group_id = os.path.join(lang, parts[2]) 
            else:
                # Structure: class/language/file.wav
                # Group by File (parts[2]) -> Unique ID
                group_id = os.path.join(lang, parts[2])
        else:
            # Structure: class/file.wav (No language folder)
            lang = "unknown"
            group_id = parts[1]
            
        if lang not in dataset_map[cls]:
            dataset_map[cls][lang] = {}
            
        if group_id not in dataset_map[cls][lang]:
            dataset_map[cls][lang][group_id] = []
            
        dataset_map[cls][lang][group_id].append(fpath)
        
    logging.info("Indexing complete. Starting Processing...")
    
    # Iterate and Split
    for cls in classes:
        for lang, groups in dataset_map[cls].items():
            # keys are speaker IDs or filenames
            keys = list(groups.keys())
            random.shuffle(keys)
            
            n_total = len(keys)
            if n_total == 0: continue
            
            n_train = int(n_total * CONFIG["train_split"])
            n_val = int(n_total * CONFIG["val_split"])
            
            # Handle small datasets gracefully
            if n_train == 0 and n_total > 0: n_train = n_total # All to train if tiny
            
            splits = {
                'train': keys[:n_train],
                'val': keys[n_train:n_train+n_val],
                'test': keys[n_train+n_val:]
            }
            
            # If Val/Test empty due to rounding, that's acceptable for small runs
            
            for split_name, split_keys in splits.items():
                for key in split_keys:
                    file_list = groups[key]
                    
                    for fpath in file_list:
                        # Output Path Reconstruction
                        # output/split/class/language/speaker/file.wav
                        
                        rel_path = os.path.relpath(fpath, raw_root)
                        save_dir = os.path.join(output_root, split_name, os.path.dirname(rel_path))
                        
                        os.makedirs(save_dir, exist_ok=True)
                        
                        fname = os.path.basename(fpath)
                        fname_no_ext = os.path.splitext(fname)[0]
                        out_name = fname_no_ext + ".wav"
                        save_path = os.path.join(save_dir, out_name)
                        
                        if os.path.exists(save_path):
                            continue 
                            
                        try:
                            processed_wav, status = process_single_file(fpath)
                            
                            if processed_wav is not None:
                                torchaudio.save(save_path, processed_wav, CONFIG["sample_rate"])
                                stats["processed"] += 1
                            else:
                                stats["discarded"] += 1
                                logging.debug(f"Discarded {fname}: {status}")
                                
                        except Exception as e:
                            stats["errors"] += 1
                            logging.error(f"Error processing {fpath}: {e}")
                            
    logging.info(f"Pipeline Finished. Stats: {stats}")
    print(f"Done. Stats: {stats}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Path to raw data root")
    parser.add_argument("--out", required=True, help="Path to output dataset root")
    args = parser.parse_args()
    
    run_pipeline(args.raw, args.out)
