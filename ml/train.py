import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .model import DualBranchModel
from .preprocessing import load_audio, trim_silence, normalize_loudness, augment_noise, pad_or_crop
from .features import extractor
import glob
import os

class VoiceDataset(Dataset):
    def __init__(self, file_list, labels, augment=False):
        self.file_list = file_list
        self.labels = labels
        self.augment = augment
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        path = self.file_list[idx]
        label = self.labels[idx]
        
        # Load & Preprocess
        waveform = load_audio(path)
        waveform = trim_silence(waveform)
        waveform = normalize_loudness(waveform)
        
        if self.augment:
            if torch.rand(1) < 0.5:
                waveform = augment_noise(waveform)
            # Add other augs here
            
        waveform = pad_or_crop(waveform, max_len_sec=4.0) # Fixed size for batching
        
        # Features
        features = extractor.extract_all(waveform)
        
        return features['mel'], features['phase'], torch.tensor(label, dtype=torch.float32)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for mel, phase, labels in loader:
        mel, phase, labels = mel.to(device), phase.to(device), labels.to(device)
        
        # Add channel dim
        mel = mel.unsqueeze(1)
        phase = phase.unsqueeze(1)
        
        optimizer.zero_grad()
        logits = model(mel, phase)
        loss = criterion(logits.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

# --- Calibration Logic ---
def calibrate_model(model, val_loader, device):
    """
    Tune temperature on validation set (frozen model).
    """
    model.eval()
    temperature = nn.Parameter(torch.ones(1).to(device))
    opt = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for mel, phase, labels in val_loader:
            mel = mel.to(device).unsqueeze(1)
            phase = phase.to(device).unsqueeze(1)
            out = model(mel, phase)
            logits_list.append(out)
            labels_list.append(labels)
            
    logits = torch.cat(logits_list).to(device)
    labels = torch.cat(labels_list).to(device)
    nll = nn.BCEWithLogitsLoss()
    
    def eval():
        opt.zero_grad()
        loss = nll(logits / temperature, labels)
        loss.backward()
        return loss
        
    opt.step(eval)
    return temperature.item()
