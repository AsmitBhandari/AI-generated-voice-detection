import requests
import base64
import numpy as np
import scipy.io.wavfile as wav
import io

def create_dummy_audio():
    sr = 16000
    t = np.linspace(0, 3, sr * 3)
    # A simple sine wave (Human-ish? No, but valid audio)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    
    byte_io = io.BytesIO()
    wav.write(byte_io, sr, (audio * 32767).astype(np.int16))
    return base64.b64encode(byte_io.getvalue()).decode('utf-8')

def test_api():
    url = "http://127.0.0.1:8000/detect-voice"
    headers = {"x-api-key": "eternity"}
    
    b64_audio = create_dummy_audio()
    
    payload = {
        "audio_base64": b64_audio,
        "language": "en"
    }
    
    try:
        res = requests.post(url, json=payload, headers=headers)
        print("Status:", res.status_code)
        print("Response:", res.json())
    except Exception as e:
        print("Failed to connect:", e)

if __name__ == "__main__":
    test_api()
