import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import base64
import io
from PIL import Image
import time

# Load the simplified TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='./models/sign_language_model_simple.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the labels dictionary
with open('./models/tensorflow_labels_dict.pickle', 'rb') as f:
    labels_dict = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

def predict_with_tflite(interpreter, input_details, output_details, data):
    """Make prediction using TensorFlow Lite model"""
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

def process_image_for_prediction(image_data):
    """Process image data and return prediction with visual data"""
    try:
        # Decode base64 image if needed
        if isinstance(image_data, str):
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            frame = image_data
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Always create an annotated frame for visual feedback
        annotated_frame = frame.copy()
        
        if not results.multi_hand_landmarks:
            # Convert annotated frame back to base64 even when no hands detected
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "error": "No hands detected in the image",
                "annotated_image": f"data:image/jpeg;base64,{annotated_image_base64}"
            }
        
        # Draw hand landmarks and bounding box
        annotated_frame = frame.copy()
        hand_data = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                annotated_frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Calculate bounding box
            h, w, _ = annotated_frame.shape
            x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
            y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            # Store hand data for prediction
            data_aux = []
            x_ = []
            y_ = []
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
            
            # Pad or truncate to expected input size
            expected_length = 42  # Based on your model input
            if len(data_aux) < expected_length:
                data_aux.extend([0] * (expected_length - len(data_aux)))
            elif len(data_aux) > expected_length:
                data_aux = data_aux[:expected_length]
            
            # Convert to numpy array and reshape
            data = np.array(data_aux).reshape(1, -1)
            
            # Make prediction
            prediction = predict_with_tflite(interpreter, input_details, output_details, data.astype(np.float32))
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
            
            # Get the predicted gesture
            predicted_gesture = labels_dict.get(str(predicted_class), f"Unknown_{predicted_class}")
            
            # Add prediction text to frame
            text = f"{predicted_gesture} ({confidence:.2f})"
            cv2.putText(annotated_frame, text, (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            hand_data.append({
                "gesture": predicted_gesture,
                "confidence": confidence,
                "predicted_class": int(predicted_class),
                "bbox": [x_min, y_min, x_max, y_max]
            })
        
        # Convert annotated frame back to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Return the best prediction (highest confidence)
        best_hand = max(hand_data, key=lambda x: x['confidence'])
        
        return {
            "gesture": best_hand["gesture"],
            "confidence": best_hand["confidence"],
            "predicted_class": best_hand["predicted_class"],
            "message": "Gesture detected successfully",
            "annotated_image": f"data:image/jpeg;base64,{annotated_image_base64}",
            "all_hands": hand_data
        }
        
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

def run_sign_language_inference():
    """Run sign language inference with webcam"""
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return {"error": "Could not open camera"}
        
        # Capture a single frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return {"error": "Failed to capture frame"}
        
        # Process the frame
        result = process_image_for_prediction(frame)
        return result
        
    except Exception as e:
        return {"error": f"Error running inference: {str(e)}"}

def start_realtime_sign_language():
    """Start real-time sign language recognition"""
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return {"error": "Could not open camera"}
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        return {
            "message": "Real-time sign language recognition started",
            "camera_ready": True
        }
        
    except Exception as e:
        return {"error": f"Error starting real-time recognition: {str(e)}"}

def process_realtime_frame(frame_data):
    """Process a single frame for real-time recognition"""
    try:
        print("🔄 Processing real-time sign language frame...")
        
        # Decode base64 image
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
        
        image_bytes = base64.b64decode(frame_data)
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process the frame
        result = process_image_for_prediction(frame)
        
        # Add timestamp
        result['timestamp'] = time.time()
        
        if 'error' not in result:
            print(f"✅ Sign language inference result: {result.get('gesture', 'Unknown')} (Confidence: {result.get('confidence', 0):.2f})")
        else:
            print(f"❌ Sign language inference error: {result['error']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in real-time sign language processing: {str(e)}")
        return {"error": f"Error processing real-time frame: {str(e)}"} 