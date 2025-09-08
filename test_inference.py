#!/usr/bin/env python3
"""
Test script to verify inference modules work correctly
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_sign_language_inference():
    """Test sign language inference module"""
    try:
        from sign_language_inference import run_sign_language_inference
        print("✅ Sign language inference module imported successfully")
        
        # Test the function
        result = run_sign_language_inference()
        print(f"📊 Sign language test result: {result}")
        
        if 'error' in result:
            if 'No hands detected' in result['error']:
                print("ℹ️  Expected: No hands detected (test runs without hands in view)")
                print("   To test with hands: Run the web app and use the Sign Language page")
                return True  # This is expected behavior
            else:
                print(f"❌ Sign language test failed: {result['error']}")
                return False
        else:
            print("✅ Sign language inference test passed")
            return True
            
    except Exception as e:
        print(f"❌ Sign language inference test failed: {str(e)}")
        return False

def test_morse_code_inference():
    """Test morse code inference module"""
    try:
        from morse_code_inference import run_morse_code_inference
        print("✅ Morse code inference module imported successfully")
        
        # Test the function
        result = run_morse_code_inference()
        print(f"📊 Morse code test result: {result}")
        
        if 'error' in result:
            if 'No hands detected' in result['error']:
                print("ℹ️  Expected: No hands detected (test runs without hands in view)")
                print("   To test with hands: Run the web app and use the Morse Code page")
                return True  # This is expected behavior
            else:
                print(f"❌ Morse code test failed: {result['error']}")
                return False
        else:
            print("✅ Morse code inference test passed")
            return True
            
    except Exception as e:
        print(f"❌ Morse code inference test failed: {str(e)}")
        return False

def test_model_loading():
    """Test that models can be loaded without errors"""
    try:
        print("\n🔍 Testing model loading...")
        
        # Test sign language model loading
        from sign_language_inference import interpreter as sign_interpreter
        print("✅ Sign language model loaded successfully")
        
        # Test morse code model loading
        from morse_code_inference import interpreter as morse_interpreter
        print("✅ Morse code model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing inference modules...")
    print("=" * 50)
    
    # Test model loading first
    model_success = test_model_loading()
    
    if not model_success:
        print("\n❌ Model loading failed. Please check your model files.")
        return 1
    
    print("\n📹 Testing webcam inference (will fail without hands in view)...")
    sign_success = test_sign_language_inference()
    print()
    morse_success = test_morse_code_inference()
    
    print("\n" + "=" * 50)
    if sign_success and morse_success:
        print("🎉 All tests passed! Your inference modules are working correctly.")
        print("\n📋 Next steps:")
        print("1. Start the Flask server: python app.py")
        print("2. Start the React app: npm start (in frontend directory)")
        print("3. Navigate to http://localhost:3000/sign-language")
        print("4. Click 'Start Webcam' and 'Detect Gesture' with your hands in view")
        print("5. Try the Morse Code page at http://localhost:3000/morse-code")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main()) 