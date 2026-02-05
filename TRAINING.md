# 🧠 Training Guide (Wav2Vec2 Pipeline)

This project uses a fine-tuned **Wav2Vec2-Base** Transformer for high-accuracy AI voice detection.

## 📋 Prerequisites

1. **Dataset**: Raw WAV files organized by class:
   ```
   data/
   ├── real/      (Human samples)
   └── fake/      (AI-generated samples)
   ```
2. **Environment**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Training the Transformer

The `ml/train_wav2vec.py` script handles automated pre-processing, feature extraction, and fine-tuning.

```bash
python ml/train_wav2vec.py
```

### Key Configurations (inside `train_wav2vec.py`):
- **Backbone**: `facebook/wav2vec2-base`
- **Epochs**: 2 (Recommended for fine-tuning on small-to-medium datasets)
- **Batch Size**: 4 (Optimized for CPU/Standard Memory)
- **Evaluation**: 10% auto-split validation

## 📦 Output Artifacts

Training produces a specialized Hugging Face-compatible directory structure:

1. **Production Model** (`models/production/model_v1.0.0`):
   - `model.safetensors`: Optimized weights.
   - `config.json`: Model hyperparameters.
   - `preprocessor_config.json`: Audio normalization settings.

## 🛠️ Deployment

Once trained, the API will automatically detect and load the production model based on the `MODEL_VERSION` environment variable.

```powershell
# Windows PowerShell
$env:MODEL_VERSION="v1.0.0"
uvicorn app.main:app

# Linux/Mac
export MODEL_VERSION=v1.0.0
uvicorn app.main:app
```

## 🧪 Verification

After training, verify the model performance using the automated scripts:

```bash
# General API Test
python scripts/verify_api.py

# Multilingual Generalization Test
python scripts/verify_multilingual.py
```
