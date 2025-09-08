#!/usr/bin/env python3
"""
Test script to verify visual feedback in sign language inference
"""

import cv2
import numpy as np
import base64
from sign_language_inference import process_image_for_prediction

def create_test_image():
    """Create a simple test image with a hand-like shape"""
    # Create a 640x480 image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simple hand-like shape (for testing purposes)
    # This is just a placeholder - in real usage, you'd use actual hand images
    cv2.circle(img, (320, 240), 50, (255, 255, 255), -1)  # Palm
    cv2.circle(img, (320, 180), 20, (255, 255, 255), -1)  # Thumb
    cv2.circle(img, (350, 200), 15, (255, 255, 255), -1)  # Index
    cv2.circle(img, (370, 220), 15, (255, 255, 255), -1)  # Middle
    cv2.circle(img, (360, 250), 15, (255, 255, 255), -1)  # Ring
    cv2.circle(img, (340, 270), 15, (255, 255, 255), -1)  # Pinky
    
    return img

def test_visual_feedback():
    """Test the visual feedback functionality"""
    print("🧪 Testing visual feedback in sign language inference...")
    
    # Create test image
    test_image = create_test_image()
    
    # Convert to base64
    _, buffer = cv2.imencode('.jpg', test_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    image_data = f"data:image/jpeg;base64,{image_base64}"
    
    print("📸 Created test image with hand-like shape")
    
    # Test the inference function
    try:
        result = process_image_for_prediction(image_data)
        
        print("🔍 Inference result:")
        print(f"   - Error: {result.get('error', 'None')}")
        print(f"   - Gesture: {result.get('gesture', 'None')}")
        print(f"   - Confidence: {result.get('confidence', 'None')}")
        print(f"   - Has annotated image: {'annotated_image' in result}")
        print(f"   - All hands data: {len(result.get('all_hands', []))} hands detected")
        
        if 'annotated_image' in result:
            print("✅ Visual feedback is working - annotated image generated!")
        else:
            print("❌ Visual feedback not working - no annotated image in result")
            
    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")

if __name__ == "__main__":
    test_visual_feedback() 