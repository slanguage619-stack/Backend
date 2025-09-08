from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from cryptography.fernet import Fernet
import os
from datetime import datetime
import json
from twilio.rest import Client
import logging
# Ensure these modules do not contain a duplicate `app = Flask(...)` initialization.
from chatbot_api import get_chatbot_response, send_chatbot_message
# Remove these imports as their functionality is now integrated via SocketIO
# from sign_language_inference import ...
# from morse_code_inference import ...

from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import pickle
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image

# Initialize Flask and Flask-SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_strong_random_secret_key_here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///signcrypt.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ---
# Encrypt/Decrypt Configuration
# ---
# Generate a consistent encryption key (store it in a secure location, e.g., an environment variable)
ENCRYPTION_KEY = os.environ.get('ENCRYPTION_KEY', Fernet.generate_key().decode()).encode()
cipher_suite = Fernet(ENCRYPTION_KEY)

# ---
# Twilio Configuration
# ---
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', 'your_account_sid')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', 'your_auth_token')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', 'your_twilio_number')

# Initialize Twilio client
try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    twilio_configured = True
except Exception as e:
    logging.warning(f"Twilio not configured: {e}")
    twilio_configured = False

# ---
# Database Models
# ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class EmergencyContact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone_number = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create database tables
with app.app_context():
    db.create_all()

# ---
# Sign Language Recognition Logic
# ---
class SignLanguageRecognizer:
    def __init__(self):
        self.MODEL_TYPE = 'simple_tflite'
        
        if self.MODEL_TYPE == 'sklearn':
            model_dict = pickle.load(open('./model.p', 'rb'))
            self.model = model_dict['model']
            self.labels_dict = {
                str(i): label for i, label in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Backspace', 'Blankspace', 'Next', 'Speak'])
            }
        else:
            try:
                self.interpreter = tf.lite.Interpreter(model_path='./models/sign_language_model_simple.tflite')
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                with open('./models/tensorflow_labels_dict.pickle', 'rb') as f:
                    self.labels_dict = pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading TFLite model or labels: {e}")
                self.interpreter = None
                self.labels_dict = {}

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        
        self.displayed_text = ""
        self.current_character = ""
        self.last_gesture = ""
        self.gesture_stable_count = 0
        self.min_stable_frames = 10
        self.next_detected = False
        self.auto_speak_enabled = True
        
        self.recent_predictions = []
        self.max_recent_predictions = 5
        
    def predict_with_tflite(self, data):
        if self.interpreter is None:
            return -1 # Return an invalid class if model isn't loaded
        
        input_data = np.asarray(data, dtype=np.float32).reshape(1, -1)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_class = np.argmax(output_data, axis=1)[0]
        return predicted_class
    
    def get_last_word(self, text):
        if text.strip():
            words = text.strip().split()
            if words:
                return words[-1]
        return ""
    
    def process_frame(self, frame):
        data_aux = []
        x_ = []
        y_ = []
        
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(frame_rgb)
        
        prediction_data = {
            'predicted_character': None,
            'stability_count': self.gesture_stable_count,
            'min_stable_frames': self.min_stable_frames,
            'current_character': self.current_character,
            'displayed_text': self.displayed_text,
            'bounding_box': None,
            'action_text': "",
            'hand_detected': False
        }
        
        if results.multi_hand_landmarks:
            prediction_data['hand_detected'] = True
            hand_landmarks = results.multi_hand_landmarks[0]
            
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())
            
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
            
            min_x, min_y = min(x_), min(y_)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)
            
            if len(data_aux) == 42:
                try:
                    prediction_key = self.predict_with_tflite(data_aux)
                    predicted_character = self.labels_dict.get(prediction_key, "Unknown")
                    
                    prediction_data['predicted_character'] = predicted_character
                    
                    if predicted_character == self.last_gesture:
                        self.gesture_stable_count += 1
                    else:
                        self.gesture_stable_count = 0
                        self.last_gesture = predicted_character
                    
                    prediction_data['stability_count'] = self.gesture_stable_count
                    
                    x1 = int(min_x * W) - 10
                    y1 = int(min_y * H) - 10
                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10
                    
                    prediction_data['bounding_box'] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                    
                    box_color = (0, 255, 0) if self.gesture_stable_count >= self.min_stable_frames else (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                    
                    stability_text = f"Stable: {self.gesture_stable_count}/{self.min_stable_frames}"
                    cv2.putText(frame, stability_text, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if self.gesture_stable_count >= self.min_stable_frames:
                        if predicted_character == "Next" and not self.next_detected:
                            self._execute_gesture_action()
                            self.next_detected = True
                        elif predicted_character != "Next":
                            self.current_character = predicted_character
                            self.next_detected = False
                    
                    if self.current_character and self.current_character != "Next":
                        action_map = {
                            "Backspace": " (say 'Next' to delete last char)",
                            "Blankspace": " (say 'Next' to add space)",
                            "Speak": " (say 'Next' to speak text)"
                        }
                        prediction_data['action_text'] = action_map.get(self.current_character, " (say 'Next' to add)")
                    
                    prediction_data['current_character'] = self.current_character
                    prediction_data['displayed_text'] = self.displayed_text
                    
                except Exception as e:
                    logging.error(f"Prediction error: {e}")
                    prediction_data['predicted_character'] = "Error"
        
        return prediction_data, frame
    
    def _add_to_recent_predictions(self, character):
        prediction_entry = {
            'character': character,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'date': datetime.now().strftime('%d/%m/%Y')
        }
        
        self.recent_predictions.insert(0, prediction_entry)
        
        if len(self.recent_predictions) > self.max_recent_predictions:
            self.recent_predictions = self.recent_predictions[:self.max_recent_predictions]
        
        socketio.emit('recent_predictions_updated', {'predictions': self.recent_predictions})
    
    def _execute_gesture_action(self):
        if self.current_character == "Backspace":
            if self.displayed_text:
                self.displayed_text = self.displayed_text[:-1]
                self._add_to_recent_predictions("Backspace")
        elif self.current_character == "Blankspace":
            if self.displayed_text and self.displayed_text[-1] != " ":
                self.displayed_text += " "
                self._add_to_recent_predictions("Space")
        elif self.current_character == "Speak":
            if self.displayed_text.strip():
                self._add_to_recent_predictions("Speak")
                socketio.emit('speak_text', {'text': self.displayed_text})
        elif self.current_character not in ["Next", "Speak"]:
            self.displayed_text += self.current_character
            self._add_to_recent_predictions(self.current_character)
    
    def clear_text(self):
        self.displayed_text = ""
    
    def clear_recent_predictions(self):
        self.recent_predictions = []
        socketio.emit('recent_predictions_updated', {'predictions': self.recent_predictions})
    
    def speak_current_text(self):
        return self.displayed_text.strip() if self.displayed_text.strip() else ""

# Initialize the recognizer
recognizer = SignLanguageRecognizer()

# ---
# HTTP Routes
# ---
@app.route('/api/register', methods=['POST'])
def register_user():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        
        if not name:
            return jsonify({'error': 'Name is required'}), 400
        
        # Check if user already exists
        existing_user = User.query.filter_by(name=name).first()
        if existing_user:
            return jsonify({'error': 'Username already exists'}), 400
        
        # Create new user
        new_user = User(name=name)
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({
            'user': {
                'id': new_user.id,
                'name': new_user.name,
                'created_at': new_user.created_at.isoformat()
            },
            'success': True
        }), 201
    
    except Exception as e:
        db.session.rollback()
        logging.error(f"Registration error: {e}")
        return jsonify({'error': 'Failed to register user'}), 500

@app.route('/api/encrypt', methods=['POST'])
def encrypt_message():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Encrypt the message
        encrypted_bytes = cipher_suite.encrypt(message.encode('utf-8'))
        encrypted_message = encrypted_bytes.decode('utf-8')
        
        return jsonify({
            'encrypted_message': encrypted_message,
            'success': True
        })
    
    except Exception as e:
        logging.error(f"Encryption error: {e}")
        return jsonify({'error': 'Failed to encrypt message'}), 500

@app.route('/api/decrypt', methods=['POST'])
def decrypt_message():
    try:
        data = request.get_json()
        encrypted_message = data.get('encrypted_message', '')
        
        if not encrypted_message:
            return jsonify({'error': 'Encrypted message is required'}), 400
        
        # Decrypt the message
        decrypted_bytes = cipher_suite.decrypt(encrypted_message.encode('utf-8'))
        decrypted_message = decrypted_bytes.decode('utf-8')
        
        return jsonify({
            'decrypted_message': decrypted_message,
            'success': True
        })
    
    except Exception as e:
        logging.error(f"Decryption error: {e}")
        return jsonify({'error': 'Failed to decrypt message. Invalid or corrupted data.'}), 500

@app.route('/api/sign-language-realtime', methods=['POST'])
def start_realtime_sign_language_recognition():
    # This route is now redundant. Real-time is started via the WebSocket client.
    return jsonify({'message': 'Use Socket.IO for real-time recognition.'}), 400

# ---
# Socket.IO Event Handlers for Real-time Sign Language Recognition
# ---
@socketio.on('connect')
def handle_connect():
    emit('status', {'msg': 'Connected to Sign Language Recognition Server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('process_frame')
def handle_frame(data):
    try:
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        prediction_data, processed_frame = recognizer.process_frame(frame)
        
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        emit('prediction_result', {
            'processed_image': f"data:image/jpeg;base64,{processed_image_b64}",
            'prediction_data': prediction_data
        })
        
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        emit('error', {'message': str(e)})

@socketio.on('clear_text')
def handle_clear_text():
    recognizer.clear_text()
    emit('text_cleared', {'text': recognizer.displayed_text})

@socketio.on('clear_recent_predictions')
def handle_clear_recent_predictions():
    recognizer.clear_recent_predictions()

@socketio.on('speak_text')
def handle_speak_text():
    text = recognizer.speak_current_text()
    if text:
        emit('speak_text', {'text': text})
    else:
        emit('no_text_to_speak')

@socketio.on('toggle_auto_speak')
def handle_toggle_auto_speak():
    recognizer.auto_speak_enabled = not recognizer.auto_speak_enabled
    status = "enabled" if recognizer.auto_speak_enabled else "disabled"
    emit('auto_speak_status', {'enabled': recognizer.auto_speak_enabled})

@app.route('/api/contacts', methods=['GET', 'OPTIONS'])
def get_contacts():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight successful'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        return response
    
    try:
        contacts = EmergencyContact.query.all()
        contacts_list = []
        for contact in contacts:
            contacts_list.append({
                'id': contact.id,
                'name': contact.name,
                'phone_number': contact.phone_number,
                'created_at': contact.created_at.isoformat()
            })
        return jsonify({'contacts': contacts_list})
    except Exception as e:
        logging.error(f"Error fetching contacts: {e}")
        return jsonify({'error': 'Failed to fetch contacts'}), 500

@app.route('/api/contacts', methods=['POST', 'OPTIONS'])
def add_contact():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight successful'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        return response
    
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        phone_number = data.get('phone_number', '').strip()
        
        if not name or not phone_number:
            return jsonify({'error': 'Name and phone number are required'}), 400
        
        # Validate phone number format (basic validation)
        if not phone_number.replace('+', '').replace(' ', '').isdigit():
            return jsonify({'error': 'Invalid phone number format'}), 400
        
        new_contact = EmergencyContact(name=name, phone_number=phone_number)
        db.session.add(new_contact)
        db.session.commit()
        
        return jsonify({
            'message': 'Contact added successfully',
            'contact': {
                'id': new_contact.id,
                'name': new_contact.name,
                'phone_number': new_contact.phone_number,
                'created_at': new_contact.created_at.isoformat()
            }
        }), 201
    
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error adding contact: {e}")
        return jsonify({'error': 'Failed to add contact'}), 500

@app.route('/api/contacts/<int:contact_id>', methods=['DELETE', 'OPTIONS'])
def delete_contact(contact_id):
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight successful'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        return response
    
    try:
        contact = EmergencyContact.query.get(contact_id)
        if not contact:
            return jsonify({'error': 'Contact not found'}), 404
        
        db.session.delete(contact)
        db.session.commit()
        
        return jsonify({'message': 'Contact deleted successfully'})
    
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting contact: {e}")
        return jsonify({'error': 'Failed to delete contact'}), 500

@app.route('/api/sos', methods=['POST', 'OPTIONS'])
def trigger_sos():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight successful'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        return response
    
    try:
        data = request.get_json()
        user_name = data.get('user_name', 'SignCrypt User')
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        if not latitude or not longitude:
            return jsonify({'error': 'GPS coordinates are required'}), 400
        
        # Import SOS functionality
        from sos_module import EmergencySOSManager, GPSCoordinates
        
        # Create SOS manager
        sos_manager = EmergencySOSManager()
        
        # Load emergency contacts from database
        contacts = EmergencyContact.query.all()
        emergency_contacts = []
        from sos_module import EmergencyContact as SOSEmergencyContact
        for contact in contacts:
            emergency_contacts.append(SOSEmergencyContact(name=contact.name, phone_number=contact.phone_number))
        
        if not emergency_contacts:
            return jsonify({'error': 'No emergency contacts configured'}), 400
        
        sos_manager.set_emergency_contacts(emergency_contacts)
        sos_manager.set_user_name(user_name)
        
        # Create GPS coordinates
        coordinates = GPSCoordinates(
            latitude=float(latitude),
            longitude=float(longitude),
            timestamp=datetime.utcnow()
        )
        
        # Trigger SOS
        result = sos_manager.trigger_sos(coordinates)
        
        return jsonify({
            'success': True,
            'message': 'SOS triggered successfully',
            'sms_results': result.get('sms_results', {}),
            'contacts_notified': len(result.get('sms_results', {}).get('details', []))
        })
        
    except Exception as e:
        logging.error(f"SOS trigger error: {e}")
        return jsonify({'error': 'Failed to trigger SOS'}), 500

@app.route('/api/chatbot', methods=['POST', 'OPTIONS'])
def chatbot_endpoint():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight successful'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        return response
    
    try:
        # Handle both JSON and multipart/form-data requests
        if request.is_json:
            data = request.get_json()
            message = data.get('message', '').strip()
            user_id = data.get('user_id', 'signcrypt_user')
            context = data.get('context', {})
            file_content = None
            file_name = None
            file_type = None
        else:
            # Handle multipart/form-data for file uploads
            message = request.form.get('message', '').strip()
            user_id = request.form.get('user_id', 'signcrypt_user')
            context = json.loads(request.form.get('context', '{}'))
            
            # Handle file upload
            file_content = None
            file_name = None
            file_type = None
            
            if 'file' in request.files:
                file = request.files['file']
                if file and file.filename:
                    file_name = file.filename
                    file_type = file.content_type
                    
                    # Read file content based on file type
                    if file_type.startswith('text/'):
                        file_content = file.read().decode('utf-8')
                    elif file_type.startswith('image/'):
                        # For images, we'll encode as base64
                        import base64
                        file_content = base64.b64encode(file.read()).decode('utf-8')
                    else:
                        # For other file types, we'll just store the filename and type
                        file_content = f"[File: {file_name} ({file_type})]"
        
        if not message and not file_content:
            return jsonify({'error': 'Message or file is required'}), 400
        
        # Import chatbot functions
        from chatbot_api import get_chatbot_response
        
        # Prepare context with file information
        enhanced_context = context.copy()
        if file_content and file_name:
            enhanced_context.update({
                'file_content': file_content,
                'file_name': file_name,
                'file_type': file_type
            })
        
        # Get response from chatbot
        response = get_chatbot_response(message or "Analyze this file", user_id, enhanced_context)
        
        return jsonify({
            'success': True,
            'response': response,
            'user_id': user_id,
            'file_processed': bool(file_content)
        })
        
    except Exception as e:
        logging.error(f"Chatbot endpoint error: {e}")
        return jsonify({'error': 'Failed to process chatbot request'}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)