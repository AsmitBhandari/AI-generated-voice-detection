import requests
import base64
import os
import time

def verify_api():
    url = "http://127.0.0.1:8001/detect-voice"
    headers = {
        "X-API-Key": "eternity",
        "Content-Type": "application/json"
    }
    
    # Test with a real human sample we downloaded
    human_sample = "data/real/english_us_0.wav"
    if not os.path.exists(human_sample):
        print(f"Error: {human_sample} not found")
        return

    with open(human_sample, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "audio_base64": audio_b64,
        "language": "en"
    }

    print(f"Testing API with {human_sample}...")
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"API Error: {e}")

    # Test with a fake sample
    fake_sample = "data/fake/english_us_1.wav"
    if os.path.exists(fake_sample):
        with open(fake_sample, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        payload["audio_base64"] = audio_b64
        print(f"\nTesting API with {fake_sample}...")
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

if __name__ == "__main__":
    verify_api()
