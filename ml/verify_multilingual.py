import os
import torch
import librosa
from datasets import Dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from tqdm import tqdm
import numpy as np

MODEL_PATH = "asmitbhandari/ai-voice-detection-multilingual"
TEST_ROOT = Path("output/test")

def verify_model():
    print(f"Loading model from {MODEL_PATH}...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    
    # Load Test Data
    print("Loading test data...")
    paths = []
    labels = []
    langs = []
    
    # Walk through test dir
    # output/test/human/english/file.wav
    for cls_name, label in [("human", 0), ("ai", 1)]:
        cls_dir = TEST_ROOT / cls_name
        if not cls_dir.exists(): continue
        
        for f in cls_dir.rglob("*.wav"):
            # Extract language
            # rel path: english/file.wav (if flattened) OR english/speaker/file.wav
            rel = f.relative_to(cls_dir)
            lang = rel.parts[0] # First folder is language
            
            paths.append(str(f))
            labels.append(label)
            langs.append(lang)
            
    print(f"Found {len(paths)} test samples across languages: {set(langs)}")
    
    # Inference
    preds = []
    
    print("Running inference...")
    with torch.no_grad():
        for path in tqdm(paths):
            # Load and preprocess
            y, sr = librosa.load(path, sr=16000)
            inputs = feature_extractor(
                y, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=16000*2, 
                truncation=True
            )
            
            logits = model(**inputs).logits
            pred_id = torch.argmax(logits, dim=-1).item()
            preds.append(pred_id)
            
    # Metrics
    overall_acc = accuracy_score(labels, preds)
    print(f"\nOverall Accuracy: {overall_acc:.4f}")
    
    print("\nPer-Language Performance:")
    lang_stats = {}
    for lang in set(langs):
        indices = [i for i, l in enumerate(langs) if l == lang]
        l_labels = [labels[i] for i in indices]
        l_preds = [preds[i] for i in indices]
        l_acc = accuracy_score(l_labels, l_preds)
        print(f"  {lang.ljust(15)}: {l_acc:.4f} ({len(indices)} samples)")
        
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Human", "AI"]))

if __name__ == "__main__":
    verify_model()
