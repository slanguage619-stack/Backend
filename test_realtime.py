#!/usr/bin/env python3
"""
Test script to verify real-time endpoints work correctly
"""

import requests
import json

def test_realtime_endpoints():
    """Test the real-time API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing real-time endpoints...")
    print("=" * 50)
    
    # Test sign language real-time start
    try:
        response = requests.post(f"{base_url}/api/sign-language-realtime")
        print(f"✅ Sign language real-time start: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Sign language real-time start failed: {e}")
    
    # Test Morse code real-time start
    try:
        response = requests.post(f"{base_url}/api/morse-code-realtime")
        print(f"✅ Morse code real-time start: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Morse code real-time start failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Real-time endpoints are working!")
    print("\n📋 Next steps:")
    print("1. Start the React app: npm start (in frontend directory)")
    print("2. Navigate to http://localhost:3000/sign-language")
    print("3. Click 'Start Webcam' then 'Start Real-time'")
    print("4. Show your hands to see real-time gesture recognition!")
    print("5. Try the Morse Code page for real-time Morse detection")

if __name__ == "__main__":
    test_realtime_endpoints() 