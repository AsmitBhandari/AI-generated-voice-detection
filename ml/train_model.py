#!/usr/bin/env python3
"""
Production-Grade Training Script for AI Voice Detection Model

Features:
- Comprehensive checkpoint management
- Resume from checkpoint
- Periodic + best model saving
- Post-training temperature calibration
- Dataset fingerprinting
- Model versioning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import sys

# Local imports
from .model import DualBranchModel
from .train import VoiceDataset, train_epoch, calibrate_model
from .checkpoint import save_checkpoint, load_checkpoint, export_production_model
from .dataset_utils import compute_dataset_fingerprint
from .features import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS


def create_dataloaders(dataset_root: str, batch_size: int, num_workers: int = 4):
    """
    Create train and validation dataloaders.
    
    Args:
        dataset_root: Root directory of processed dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    import glob
    
    dataset_path = Path(dataset_root)
    
    # Collect training files
    train_human = glob.glob(str(dataset_path / "train" / "human" / "**" / "*.wav"), recursive=True)
    train_ai = glob.glob(str(dataset_path / "train" / "ai" / "**" / "*.wav"), recursive=True)
    
    train_files = train_human + train_ai
    train_labels = [0] * len(train_human) + [1] * len(train_ai)  # 0=human, 1=ai
    
    # Collect validation files
    val_human = glob.glob(str(dataset_path / "val" / "human" / "**" / "*.wav"), recursive=True)
    val_ai = glob.glob(str(dataset_path / "val" / "ai" / "**" / "*.wav"), recursive=True)
    
    val_files = val_human + val_ai
    val_labels = [0] * len(val_human) + [1] * len(val_ai)
    
    print(f"Training samples: {len(train_files)} (Human: {len(train_human)}, AI: {len(train_ai)})")
    print(f"Validation samples: {len(val_files)} (Human: {len(val_human)}, AI: {len(val_ai)})")
    
    if len(train_files) == 0:
        raise ValueError(f"No training files found in {dataset_root}/train/")
    
    # Create datasets
    train_dataset = VoiceDataset(train_files, train_labels, augment=True)
    val_dataset = VoiceDataset(val_files, val_labels, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def validate(model, val_loader, criterion, device, temperature=1.0):
    """
    Validate model on validation set.
    
    Args:
        model: PyTorch model
        val_loader: Validation dataloader
        criterion: Loss function
        device: Device to run on
        temperature: Temperature scaling (default 1.0 during training)
    
    Returns:
        Tuple of (val_loss, val_accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for mel, phase, labels in val_loader:
            mel, phase, labels = mel.to(device), phase.to(device), labels.to(device)
            
            # Add channel dim
            mel = mel.unsqueeze(1)
            phase = phase.unsqueeze(1)
            
            # Forward pass
            logits = model(mel, phase)
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            
            # Loss
            loss = criterion(scaled_logits.squeeze(), labels)
            total_loss += loss.item()
            
            # Accuracy
            probs = torch.sigmoid(scaled_logits.squeeze())
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train AI Voice Detection Model")
    
    # Dataset
    parser.add_argument("--dataset", type=str, required=True, help="Path to processed dataset root")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    
    # Checkpointing
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    
    # Model versioning
    parser.add_argument("--version", type=str, default="v1.0.0", help="Model version (semantic versioning)")
    
    # System
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    print("="*60)
    print("AI Voice Detection - Training Script")
    print("="*60)
    print(f"Model Version: {args.version}")
    print(f"Device: {args.device}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print("="*60)
    
    # Device
    device = torch.device(args.device)
    
    # 1. Dataset Fingerprinting
    print("\n[1/7] Computing dataset fingerprint...")
    dataset_metadata = compute_dataset_fingerprint(args.dataset)
    
    # 2. Create Dataloaders
    print("\n[2/7] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        args.dataset,
        args.batch_size,
        args.num_workers
    )
    
    # 3. Model, Optimizer, Scheduler
    print("\n[3/7] Initializing model...")
    model = DualBranchModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # Training configuration
    training_config = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.epochs,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "feature_config": {
            "sample_rate": SAMPLE_RATE,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS
        }
    }
    
    # 4. Resume from Checkpoint (Optional)
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\n[4/7] Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device=args.device
        )
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"  Resuming from epoch {start_epoch}")
    else:
        print("\n[4/7] Starting training from scratch")
    
    # 5. Training Loop
    print(f"\n[5/7] Training for {args.epochs} epochs...")
    print("-"*60)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, temperature=1.0)
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save Periodic Checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = f"models/checkpoints/epoch_{epoch+1}.pth"
            save_checkpoint(
                path=checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                temperature=1.0,  # Always 1.0 during training
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_acc,
                dataset_metadata=dataset_metadata,
                training_config=training_config,
                model_version=args.version
            )
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✓ New best model! (Val Loss: {val_loss:.4f})")
            save_checkpoint(
                path="models/checkpoints/best_model.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                temperature=1.0,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_acc,
                dataset_metadata=dataset_metadata,
                training_config=training_config,
                model_version=args.version
            )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # 6. Temperature Calibration
    print("\n[6/7] Calibrating temperature on validation set...")
    best_checkpoint = torch.load("models/checkpoints/best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    calibrated_temp = calibrate_model(model, val_loader, device)
    print(f"  ✓ Calibrated temperature: {calibrated_temp:.4f}")
    
    # Update checkpoint with calibrated temperature
    best_checkpoint['temperature'] = calibrated_temp
    torch.save(best_checkpoint, "models/checkpoints/best_model_calibrated.pth")
    print(f"  ✓ Saved calibrated checkpoint: models/checkpoints/best_model_calibrated.pth")
    
    # 7. Export Production Model
    print(f"\n[7/7] Exporting production model...")
    production_path = f"models/production/model_{args.version}.pth"
    export_production_model(
        checkpoint_path="models/checkpoints/best_model_calibrated.pth",
        output_path=production_path,
        languages=dataset_metadata.get("languages")
    )
    
    print("\n" + "="*60)
    print("✓ Training pipeline complete!")
    print(f"✓ Production model: {production_path}")
    print(f"✓ Model version: {args.version}")
    print(f"✓ Temperature: {calibrated_temp:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
