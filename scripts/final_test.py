import requests
import base64
import os

url = "http://127.0.0.1:8100/detect-voice"
headers = {
    "X-API-Key": "eternity",
    "Content-Type": "application/json"
}

def run_test(name, payload):
    print(f"\n--- Testing {name} ---")
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Result: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

# 1. Human (Local)
with open("data/real/english_us_0.wav", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")
run_test("Human (Local English)", {"audio_base64": b64, "language": "en"})

# 2. AI (Local)
with open("data/fake/english_us_1.wav", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")
run_test("AI (Local English)", {"audio_base64": b64, "language": "en"})

# 3. Public URL (Human)
public_url = "https://www.w3schools.com/html/horse.mp3"
run_test("Public URL (Human)", {"audio_url": public_url, "language": "en"})
