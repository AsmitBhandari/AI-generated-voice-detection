
# Data Processing Configuration

CONFIG = {
    "sample_rate": 16000,
    "target_format": "wav",
    "target_db": -20.0,      # Desired RMS loudness 
    "min_duration": 2.0,     # Discard if shorter
    "max_duration": 6.0,     # Crop if longer
    "silence_top_db": 30,    # Silence threshold (lower = more sensitive to noise)
    "train_split": 0.8,
    "val_split": 0.1,        # Test is remainder (0.1)
    "random_seed": 42
}
