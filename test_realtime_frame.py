#!/usr/bin/env python3
"""
Test script to simulate real-time frame processing
"""

import requests
import base64
import cv2
import numpy as np
import time

def create_test_frame():
    """Create a test frame with a hand-like shape"""
    # Create a 640x480 image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simple hand-like shape (for testing purposes)
    cv2.circle(img, (320, 240), 50, (255, 255, 255), -1)  # Palm
    cv2.circle(img, (320, 180), 20, (255, 255, 255), -1)  # Thumb
    cv2.circle(img, (350, 200), 15, (255, 255, 255), -1)  # Index
    cv2.circle(img, (370, 220), 15, (255, 255, 255), -1)  # Middle
    cv2.circle(img, (360, 250), 15, (255, 255, 255), -1)  # Ring
    cv2.circle(img, (340, 270), 15, (255, 255, 255), -1)  # Pinky
    
    return img

def test_realtime_frame_processing():
    """Test the real-time frame processing API"""
    print("🧪 Testing real-time frame processing...")
    
    # Create test frame
    test_frame = create_test_frame()
    
    # Convert to base64 (simulating frontend)
    _, buffer = cv2.imencode('.jpg', test_frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    frame_data = f"data:image/jpeg;base64,{image_base64}"
    
    print("📸 Created test frame")
    
    # Test the API endpoint
    try:
        response = requests.post(
            'http://localhost:5000/api/sign-language-realtime-frame',
            json={'frame': frame_data},
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"📡 API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response:")
            print(f"   - Error: {data.get('error', 'None')}")
            print(f"   - Gesture: {data.get('gesture', 'None')}")
            print(f"   - Confidence: {data.get('confidence', 'None')}")
            print(f"   - Has annotated image: {'annotated_image' in data}")
            print(f"   - All hands data: {len(data.get('all_hands', []))} hands detected")
            
            if 'annotated_image' in data:
                print("✅ Visual feedback is working!")
            else:
                print("❌ No annotated image in response")
        else:
            print(f"❌ API Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing API: {str(e)}")

def test_multiple_frames():
    """Test multiple frames to simulate real-time processing"""
    print("\n🔄 Testing multiple frames (simulating real-time)...")
    
    for i in range(5):
        print(f"\n📸 Frame {i+1}/5:")
        
        # Create test frame
        test_frame = create_test_frame()
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', test_frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        frame_data = f"data:image/jpeg;base64,{image_base64}"
        
        try:
            response = requests.post(
                'http://localhost:5000/api/sign-language-realtime-frame',
                json={'frame': frame_data},
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'error' in data:
                    print(f"   ❌ {data['error']}")
                else:
                    print(f"   ✅ Gesture: {data.get('gesture', 'Unknown')} (Confidence: {data.get('confidence', 0):.2f})")
            else:
                print(f"   ❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        time.sleep(0.3)  # Simulate 300ms interval

if __name__ == "__main__":
    test_realtime_frame_processing()
    test_multiple_frames() 