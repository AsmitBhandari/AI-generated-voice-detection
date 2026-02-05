#!/usr/bin/env python3
"""
Test script for buildathon API with explanation validation.
"""

import requests
import json

# Configuration
API_URL = "http://localhost:8000/detect-voice"
API_KEY = "eternity"

# Test audio URL
TEST_URL = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"

def test_explanation_format():
    """Test API explanation format."""
    print("="*60)
    print("Testing Explanation Format")
    print("="*60)
    
    payload = {
        "audio_url": TEST_URL,
        "language": "en"
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        
        print(f"Status Code: {response.status_code}\n")
        
        data = response.json()
        print("Response JSON:")
        print(json.dumps(data, indent=2))
        
        # Validate explanation structure
        if "error" in data:
            print(f"\n❌ Error: {data['error']}")
            return
        
        print("\n" + "="*60)
        print("Validation Results")
        print("="*60)
        
        # Check required fields
        required_fields = ["prediction", "confidence", "explanation"]
        for field in required_fields:
            if field in data:
                print(f"✅ {field}: present")
            else:
                print(f"❌ {field}: MISSING")
        
        # Check explanation structure
        if "explanation" in data:
            explanation = data["explanation"]
            
            if "summary" in explanation:
                print(f"✅ explanation.summary: present")
                print(f"   Length: {len(explanation['summary'])} chars")
            else:
                print(f"❌ explanation.summary: MISSING")
            
            if "signals" in explanation:
                print(f"✅ explanation.signals: present")
                
                # Check signal details
                for signal_name in ["spectral_entropy", "temporal_consistency"]:
                    if signal_name in explanation["signals"]:
                        signal = explanation["signals"][signal_name]
                        print(f"\n   {signal_name}:")
                        print(f"     - value: {signal.get('value', 'MISSING')}")
                        print(f"     - interpretation: {len(signal.get('interpretation', ''))} chars")
                        print(f"     - indicator: {signal.get('indicator', 'MISSING')}")
                    else:
                        print(f"   ❌ {signal_name}: MISSING")
            else:
                print(f"❌ explanation.signals: MISSING")
        
        print("\n" + "="*60)
        print("Human-Readable Summary")
        print("="*60)
        if "explanation" in data and "summary" in data["explanation"]:
            print(data["explanation"]["summary"])
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_explanation_format()
