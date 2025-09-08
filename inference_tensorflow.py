import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import threading

# Choose which model to use: 'h5', 'tflite', 'simple_tflite', or 'sklearn'
MODEL_TYPE = 'simple_tflite'  # Change this to test different models

if MODEL_TYPE == 'sklearn':
    # Load the original sklearn model
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    
    labels_dict = {
        '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E',
        '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J',
        '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O',
        '15': 'P', '16': 'Q', '17': 'R', '18': 'S', '19': 'T',
        '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y',
        '25': 'Z',
        '26': '0', '27': '1', '28': '2', '29': '3', '30': '4',
        '31': '5', '32': '6', '33': '7', '34': '8', '35': '9',
        '36': 'Backspace',
        '37': 'Blankspace',
        '38': 'Next',
        '39': 'Speak'
    }
    
elif MODEL_TYPE == 'h5':
    # Load the TensorFlow Keras model
    model = tf.keras.models.load_model('./sign_language_model.h5')
    
    # Load the labels dictionary
    with open('./tensorflow_labels_dict.pickle', 'rb') as f:
        labels_dict = pickle.load(f)
    
elif MODEL_TYPE == 'tflite':
    # Load the original TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path='./sign_language_model.tflite')
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load the labels dictionary
    with open('./tensorflow_labels_dict.pickle', 'rb') as f:
        labels_dict = pickle.load(f)

elif MODEL_TYPE == 'simple_tflite':
    # Load the high-accuracy simplified TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path='./sign_language_model_simple.tflite')
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load the labels dictionary
    with open('./tensorflow_labels_dict.pickle', 'rb') as f:
        labels_dict = pickle.load(f)

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Configure TTS engine for macOS
try:
    voices = tts_engine.getProperty('voices')
    if voices:
        # Look for Samantha (best quality English voice) first
        samantha_voice = None
        for voice in voices:
            if 'Samantha' in voice.name:
                samantha_voice = voice
                break
        
        if samantha_voice:
            tts_engine.setProperty('voice', samantha_voice.id)
            print(f"🎤 Using high-quality voice: {samantha_voice.name}")
        else:
            # Fallback to first available voice
            tts_engine.setProperty('voice', voices[0].id)
            print(f"🎤 Using voice: {voices[0].name}")
    
    tts_engine.setProperty('rate', 150)  # Speed of speech
    tts_engine.setProperty('volume', 1.0)  # Maximum volume level
    print("✅ TTS engine initialized successfully")
except Exception as e:
    print(f"⚠️ TTS engine initialization warning: {e}")

def speak_text(text):
    """Function to speak text in a separate thread"""
    global speech_active
    if text.strip() and not speech_active:  # Only speak if there's text and no active speech
        try:
            speech_active = True
            print(f"🔊 Speaking: '{text}'")
            tts_engine.say(text)
            tts_engine.runAndWait()
            print("✅ Speech completed")
        except Exception as e:
            print(f"❌ Speech error: {e}")
        finally:
            speech_active = False
    elif speech_active:
        print("⏳ Speech already in progress, skipping...")
    else:
        print("📝 No text to speak")

def get_last_word(text):
    """Extract the last word from text"""
    if text.strip():
        words = text.strip().split()
        if words:
            return words[-1]
    return ""

def auto_speak_word(word):
    """Automatically speak a completed word"""
    global auto_speak_enabled
    if auto_speak_enabled and word and not speech_active:
        print(f"🤖 Auto-announcing word: '{word}'")
        speech_thread = threading.Thread(target=speak_text, args=(word,))
        speech_thread.daemon = True
        speech_thread.start()

# Text accumulation variables
displayed_text = ""
current_character = ""
last_gesture = ""
gesture_stable_count = 0
min_stable_frames = 10  # Gesture must be stable for this many frames
next_detected = False
speech_active = False  # Flag to prevent multiple speech instances
auto_speak_enabled = True  # Flag to enable/disable automatic word announcement

# Mouse callback function for button clicks
def mouse_callback(event, x, y, flags, param):
    global displayed_text
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is on Clear button area
        if clear_button_x <= x <= clear_button_x + clear_button_width and clear_button_y <= y <= clear_button_y + clear_button_height:
            displayed_text = ""
            print("Cleared all text via button click")
        
        # Check if click is on Speak button area
        elif speak_button_x <= x <= speak_button_x + speak_button_width and speak_button_y <= y <= speak_button_y + speak_button_height:
            # Speak all text when speak button is clicked
            if displayed_text.strip() and not speech_active:
                print(f"🎤 Manual speak button clicked")
                speech_thread = threading.Thread(target=speak_text, args=(displayed_text,))
                speech_thread.daemon = True
                speech_thread.start()
            elif speech_active:
                print("⏳ Speech already in progress")
            else:
                print("📝 No text to speak")

# Clear button properties
clear_button_width = 80
clear_button_height = 30
clear_button_x = 0  # Will be set dynamically
clear_button_y = 85  # Below the text area

# Speak button properties
speak_button_width = 80
speak_button_height = 30
speak_button_x = 0  # Will be set dynamically
speak_button_y = 85  # Same row as clear button

print(f"Using {MODEL_TYPE} model")
print(f"Labels: {labels_dict}")

cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera with index 0")
    # Try index 1 as backup
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open any camera")
        exit()
    else:
        print("Using camera index 1")
else:
    print("Using camera index 0")

# Set up mouse callback for the window
cv2.namedWindow('Sign Language Recognition')
cv2.setMouseCallback('Sign Language Recognition', mouse_callback)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

def predict_with_sklearn(model, data):
    """Predict using sklearn model"""
    prediction = model.predict([np.asarray(data)])
    return str(prediction[0])

def predict_with_h5(model, data):
    """Predict using Keras .h5 model"""
    data_array = np.asarray(data).reshape(1, -1)
    prediction = model.predict(data_array, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

def predict_with_tflite(interpreter, input_details, output_details, data):
    """Predict using TensorFlow Lite model"""
    # Prepare input data
    input_data = np.asarray(data, dtype=np.float32).reshape(1, -1)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=1)[0]
    return predicted_class

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Failed to capture frame from camera")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
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

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Only make prediction if we have the expected number of features (42)
        if len(data_aux) == 42:
            try:
                if MODEL_TYPE == 'sklearn':
                    prediction_key = predict_with_sklearn(model, data_aux)
                    predicted_character = labels_dict[prediction_key]
                    
                elif MODEL_TYPE == 'h5':
                    prediction_key = predict_with_h5(model, data_aux)
                    predicted_character = labels_dict[prediction_key]
                    
                elif MODEL_TYPE in ['tflite', 'simple_tflite']:
                    prediction_key = predict_with_tflite(interpreter, input_details, output_details, data_aux)
                    predicted_character = labels_dict[prediction_key]

                # Handle gesture stability and text accumulation
                if predicted_character == last_gesture:
                    gesture_stable_count += 1
                else:
                    gesture_stable_count = 0
                    last_gesture = predicted_character
                
                # If gesture is stable for enough frames
                if gesture_stable_count >= min_stable_frames:
                    if predicted_character == "Next" and not next_detected:
                        # Execute action based on current character when "Next" is detected
                        if current_character == "Backspace":
                            # Remove last character when backspace + next is detected
                            if displayed_text:
                                old_word = get_last_word(displayed_text)
                                displayed_text = displayed_text[:-1]
                                new_word = get_last_word(displayed_text)
                                print(f"Removed last character. Current text: '{displayed_text}'")
                                # Auto-speak if word changed significantly
                                if old_word != new_word and len(new_word) >= 3:
                                    auto_speak_word(new_word)
                        elif current_character == "Blankspace":
                            # Add space when blankspace + next is detected - this completes a word
                            if displayed_text and displayed_text[-1] != " ":  # Avoid double spaces
                                completed_word = get_last_word(displayed_text)
                                displayed_text += " "
                                print(f"Added space. Current text: '{displayed_text}'")
                                # Auto-speak the completed word
                                if completed_word and len(completed_word) >= 2:
                                    auto_speak_word(completed_word)
                        elif current_character == "Speak":
                            # Speak the current text when speak + next is detected
                            if displayed_text.strip() and not speech_active:
                                print(f"🎯 Speak gesture triggered!")
                                # Run speech in a separate thread to avoid blocking
                                speech_thread = threading.Thread(target=speak_text, args=(displayed_text,))
                                speech_thread.daemon = True
                                speech_thread.start()
                            elif speech_active:
                                print("⏳ Speech already in progress")
                            else:
                                print("📝 No text to speak via gesture")
                        elif current_character and current_character not in ["Next", "Speak"]:
                            # Add current character to displayed text
                            displayed_text += current_character
                            print(f"Added '{current_character}' to text. Current text: '{displayed_text}'")
                        next_detected = True
                    elif predicted_character not in ["Next"]:
                        # Update current character (but don't execute action until "Next")
                        if predicted_character == "Speak" and current_character != "Speak":
                            print(f"🎤 Speak gesture detected - waiting for 'Next' to execute")
                        current_character = predicted_character
                        next_detected = False
                
                # Draw bounding box (green for stable gestures, yellow for unstable)
                box_color = (0, 255, 0) if gesture_stable_count >= min_stable_frames else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
                
                # Display current prediction
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3,
                            cv2.LINE_AA)
                
                # Display stability indicator
                stability_text = f"Stable: {gesture_stable_count}/{min_stable_frames}"
                cv2.putText(frame, stability_text, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA)
                            
            except Exception as e:
                print(f"Prediction error: {e}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, "Error", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3,
                            cv2.LINE_AA)
        else:
            # Draw bounding box but no prediction for inconsistent data
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, "Multiple hands", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3,
                        cv2.LINE_AA)
            gesture_stable_count = 0
            last_gesture = ""

    # Display the accumulated text on screen
    text_bg_height = 80
    cv2.rectangle(frame, (0, 0), (W, text_bg_height), (0, 0, 0), -1)  # Black background
    
    # Display accumulated text
    cv2.putText(frame, f"Text: {displayed_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display current character waiting to be added
    if current_character and current_character not in ["Next"]:
        action_text = ""
        if current_character == "Backspace":
            action_text = " (say 'Next' to delete last char)"
        elif current_character == "Blankspace":
            action_text = " (say 'Next' to add space)"
        elif current_character == "Speak":
            action_text = " (say 'Next' to speak text)"
        else:
            action_text = " (say 'Next' to add)"
        cv2.putText(frame, f"Current: {current_character}{action_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Draw Speak button (blue button, left of clear button)
    speak_button_x = W - (speak_button_width + clear_button_width + 20)  # Leave space between buttons
    cv2.rectangle(frame, (speak_button_x, speak_button_y), (speak_button_x + speak_button_width, speak_button_y + speak_button_height), (255, 140, 0), -1)  # Orange button
    cv2.rectangle(frame, (speak_button_x, speak_button_y), (speak_button_x + speak_button_width, speak_button_y + speak_button_height), (255, 255, 255), 2)  # White border
    cv2.putText(frame, "SPEAK", (speak_button_x + 8, speak_button_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Draw Clear button (red button, on the right)
    clear_button_x = W - clear_button_width - 10  # Position on the right side
    cv2.rectangle(frame, (clear_button_x, clear_button_y), (clear_button_x + clear_button_width, clear_button_y + clear_button_height), (0, 0, 255), -1)  # Red button
    cv2.rectangle(frame, (clear_button_x, clear_button_y), (clear_button_x + clear_button_width, clear_button_y + clear_button_height), (255, 255, 255), 2)  # White border
    cv2.putText(frame, "CLEAR", (clear_button_x + 10, clear_button_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display instructions
    auto_status = "ON" if auto_speak_enabled else "OFF"
    cv2.putText(frame, f"Instructions: Gesture->Next | Backspace=delete | Space=complete word | Auto-speak:{auto_status} | Press 'A' to toggle | Click SPEAK/CLEAR", 
                (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Sign Language Recognition', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a') or key == ord('A'):
        # Toggle auto-speak with 'A' key
        auto_speak_enabled = not auto_speak_enabled
        status = "enabled" if auto_speak_enabled else "disabled"
        print(f"🔄 Auto-speak {status}")

cap.release()
cv2.destroyAllWindows()
