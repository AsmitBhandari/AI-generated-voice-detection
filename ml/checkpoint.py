import torch
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def save_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    temperature: float,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    dataset_metadata: Dict[str, Any],
    training_config: Dict[str, Any],
    model_version: str = "v1.0.0"
):
    """
    Save a comprehensive training checkpoint.
    
    Args:
        path: Path to save checkpoint
        epoch: Current epoch number
        model: PyTorch model
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        temperature: Temperature value (1.0 during training)
        train_loss: Training loss for this epoch
        val_loss: Validation loss for this epoch
        val_accuracy: Validation accuracy for this epoch
        dataset_metadata: Dataset fingerprint and statistics
        training_config: Training hyperparameters and feature config
        model_version: Semantic version string
    """
    checkpoint = {
        # Training State
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        
        # Calibration
        "temperature": temperature,
        
        # Metrics
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        
        # Training Configuration
        "training_config": training_config,
        
        # Dataset Metadata
        "dataset_metadata": dataset_metadata,
        
        # Versioning
        "model_version": model_version,
        "timestamp": datetime.now().isoformat(),
    }
    
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to map tensors to
    
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    print(f"✓ Checkpoint loaded: {path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Temperature: {checkpoint['temperature']}")
    print(f"  Version: {checkpoint['model_version']}")
    
    return checkpoint


def export_production_model(
    checkpoint_path: str,
    output_path: str,
    languages: Optional[list] = None
):
    """
    Export a production-ready model artifact from a training checkpoint.
    
    Strips optimizer and scheduler states, keeping only:
    - Model weights
    - Calibrated temperature
    - Minimal metadata
    
    Args:
        checkpoint_path: Path to training checkpoint
        output_path: Path to save production model
        languages: Optional list of supported languages
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract feature config from training config
    feature_config = checkpoint["training_config"].get("feature_config", {
        "sample_rate": 16000,
        "n_fft": 1024,
        "hop_length": 256,
        "n_mels": 80
    })
    
    production_checkpoint = {
        # Model Only
        "model_state_dict": checkpoint["model_state_dict"],
        
        # Calibration
        "temperature": checkpoint["temperature"],
        
        # Minimal Metadata
        "model_version": checkpoint["model_version"],
        "timestamp": datetime.now().isoformat(),
        "languages": languages or checkpoint["dataset_metadata"].get("languages", []),
        "feature_config": feature_config,
        "dataset_hash": checkpoint["dataset_metadata"]["dataset_hash"],
    }
    
    torch.save(production_checkpoint, output_path)
    
    # Size comparison
    import os
    orig_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    prod_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"✓ Production model exported: {output_path}")
    print(f"  Original checkpoint: {orig_size:.1f} MB")
    print(f"  Production artifact: {prod_size:.1f} MB")
    print(f"  Size reduction: {((orig_size - prod_size) / orig_size * 100):.1f}%")
