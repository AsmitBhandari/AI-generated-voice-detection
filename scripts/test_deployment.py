import requests
import sys
import json

def test_deployment(base_url):
    print(f"Testing Deployment at: {base_url}")
    
    # 1. Health Check
    try:
        health_res = requests.get(f"{base_url}/health")
        print("\n[1] Health Check:")
        print(json.dumps(health_res.json(), indent=2))
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    # 2. Prediction Test (Using Hackathon Schema)
    detect_url = f"{base_url}/predict"
    payload = {
        "language": "en",
        "audio_format": "mp3",
        "audio_base64": "" # Will be filled below
    }
    
    # Get actual base64 of a small sample
    import base64
    try:
        # Use a stable public URL etc or just a dummy small file
        # For the test script, we'll download one and convert
        res = requests.get("https://www.w3schools.com/html/horse.mp3")
        payload["audio_base64"] = base64.b64encode(res.content).decode("utf-8")
    except:
        print("Failed to prepare test audio.")
        return
    headers = {
        "X-API-Key": "eternity",
        "Content-Type": "application/json"
    }
    
    print("\n[2] Prediction Test (Voice Detection):")
    try:
        res = requests.post(detect_url, json=payload, headers=headers)
        if res.status_code == 200:
            print("SUCCESS! Response:")
            print(json.dumps(res.json(), indent=2))
        else:
            print(f"FAILED (Status {res.status_code}):")
            print(res.text)
    except Exception as e:
        print(f"Prediction test failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_deployment.py <YOUR_RAILWAY_URL>")
        print("Example: python scripts/test_deployment.py https://my-app.up.railway.app")
    else:
        url = sys.argv[1].rstrip("/")
        test_deployment(url)
