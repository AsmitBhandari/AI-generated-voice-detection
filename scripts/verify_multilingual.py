import requests
import base64
import os
import glob
from pathlib import Path

def verify_multilingual():
    url = "http://127.0.0.1:8001/detect-voice"
    headers = {
        "X-API-Key": "eternity",
        "Content-Type": "application/json"
    }
    
    test_dir = Path("data/multilingual_test_indian")
    langs = ["hindi", "tamil", "telugu", "malayalam"]
    
    results = {lang: {"correct": 0, "total": 0} for lang in langs}
    
    # Test Human Files
    for lang in langs:
        files = list(test_dir.glob(f"{lang}_human_*.wav"))
        for fpath in files:
            with open(fpath, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            
            payload = {"audio_base64": audio_b64, "language": "en"}
            try:
                response = requests.post(url, json=payload, headers=headers)
                data = response.json()
                if data.get("prediction") == "human":
                    results[lang]["correct"] += 1
                results[lang]["total"] += 1
            except Exception as e:
                print(f"Error testing {fpath}: {e}")

    # Test AI Files
    for lang in langs:
        files = list(test_dir.glob(f"{lang}_ai_*.wav"))
        for fpath in files:
            with open(fpath, f"rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            
            payload = {"audio_base64": audio_b64, "language": "en"}
            try:
                response = requests.post(url, json=payload, headers=headers)
                data = response.json()
                if data.get("prediction") == "ai_generated":
                    results[lang]["correct"] += 1
                results[lang]["total"] += 1
            except Exception as e:
                print(f"Error testing {fpath}: {e}")

    print("\n=== MULTILINGUAL GENERALIZATION RESULTS ===")
    overall_correct = 0
    overall_total = 0
    for lang, res in results.items():
        acc = (res["correct"] / res["total"]) * 100 if res["total"] > 0 else 0
        print(f"{lang.capitalize()}: {res['correct']}/{res['total']} ({acc:.1f}%)")
        overall_correct += res["correct"]
        overall_total += res["total"]
    
    overall_acc = (overall_correct / overall_total) * 100 if overall_total > 0 else 0
    print(f"\nOVERALL ACCURACY: {overall_acc:.1f}%")

if __name__ == "__main__":
    verify_multilingual()
