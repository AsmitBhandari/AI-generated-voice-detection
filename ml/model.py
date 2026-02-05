import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import librosa
import numpy as np

class ModelWrapper:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_version = "v1.0.0-wav2vec2"
        
        if model_path:
            self.load(model_path)
        else:
            # Default to our production model on Hugging Face
            print("No path provided. loading from Hugging Face Hub...")
            self.load("asmitbhandari/ai-voice-detection-multilingual")
        
        self.model.eval()

    def predict_from_waveform(self, waveform):
        """
        Accepts a numpy array (1D) and returns confidence of 'ai_generated'.
        """
        with torch.no_grad():
            inputs = self.feature_extractor(
                waveform, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            
            # Label 0: Human, Label 1: AI (as defined in train_wav2vec.py)
            ai_gen_prob = probs[0][1].item()
            return ai_gen_prob

    def load(self, path):
        """
        Load fine-tuned Wav2Vec2 model.
        """
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(path)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(path).to(self.device)
        print(f"Wav2Vec2 Model loaded from: {path}")
    
    def save_production(self, path, version, dataset_hash, languages=None):
        """
        Save production model artifact.
        
        Args:
            path: Output path for production model
            version: Model version string (e.g., "v1.0.0")
            dataset_hash: Dataset fingerprint hash
            languages: List of supported languages
        """
        from datetime import datetime
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "temperature": self.temperature,
            "model_version": version,
            "timestamp": datetime.now().isoformat(),
            "dataset_hash": dataset_hash,
            "languages": languages or [],
            "feature_config": self.feature_config or {
                "sample_rate": 16000,
                "n_fft": 1024,
                "hop_length": 256,
                "n_mels": 80
            }
        }
        
        torch.save(checkpoint, path)
        print(f"Production model saved: {path}")

