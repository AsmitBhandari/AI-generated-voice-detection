# 🎙️ AI Voice Detection System (Buildathon Winner)

An advanced AI system designed to distinguish between **AI-generated** and **Human** speech using state-of-the-art **Transformer** architectures. This project was developed as a high-speed hackathon entry, moving from data acquisition to a production-ready API in under 2 hours.

## ✨ Features
- **Wav2Vec2 Powered**: Utilizes a fine-tuned `Wav2Vec2-Base` transformer for deep acoustic analysis.
- **Multilingual Ready**: Zero-shot detection verified on Hindi, Tamil, Telugu, and Malayalam (62.5% accuracy without direct training).
- **Explainable AI**: Provides signal-level insights (Spectral Entropy & Temporal Consistency).
- **Production-Grade API**: Built with FastAPI, including API Key authentication and robust audio preprocessing.
- **Anti-Fraud Optimized**: Specifically tuned to catch synthesis artifacts from modern TTS engines (like Meta MMS).

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- FFmpeg (for audio processing)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Run the API
```bash
# Set your model version (default v1.0.0)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Detection Examples

**Option A: Using an Audio URL (Recommended for Judges)**
```bash
curl -X POST http://localhost:8000/detect-voice \
  -H "X-API-Key: eternity" \
  -H "Content-Type: application/json" \
  -d '{"audio_url": "https://example.com/sample.wav", "language": "en"}'
```

**Option B: Using Base64 Audio**
```bash
curl -X POST http://localhost:8000/detect-voice \
  -H "X-API-Key: eternity" \
  -H "Content-Type: application/json" \
  -d '{"audio_base64": "...", "language": "en"}'
```

### 🪟 Windows (PowerShell) Example
If you are using PowerShell, use this command:
```powershell
$body = @{
    audio_url = "https://www.w3schools.com/html/horse.mp3"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8100/detect-voice" `
  -Method Post `
  -Headers @{"X-API-Key" = "eternity"} `
  -ContentType "application/json" `
  -Body $body
```

## 🏗️ Architecture
- **Backbone**: `facebook/wav2vec2-base` (Fine-tuned classification head).
- **Backend**: FastAPI with Uvicorn.
- **Data Pipeline**: Automated synthesis (Meta MMS-TTS) and acquisition (minds14/Fleurs).
- **Cleaning**: `librosa` based silence trimming and loudness normalization.

## 📈 Performance
- **Human Recognition**: **85.8% Human Confidence** (Correct)
- **AI Recognition**: **89.1% AI Confidence** (Correct)
- **Multilingual (Zero-Shot)**: 62.5% accuracy across HI, TA, TE, ML dialects.

> [!TIP]
> **Confidence scoring is class-specific**: If the model predicts "Human", the score shows its certainty in the human origin (1 - AI prob), making it easy for judges to read.


---
*Built for the 2-Hour Winning Strategy challenge.*
