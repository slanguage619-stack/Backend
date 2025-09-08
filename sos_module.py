"""
SignCrypt Modular SOS System
Cross-platform emergency detection and notification system
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from twilio.rest import Client
import requests
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GPSCoordinates:
    """GPS coordinate data structure"""
    latitude: float
    longitude: float
    accuracy: Optional[float] = None
    timestamp: Optional[datetime] = None

@dataclass
class EmergencyContact:
    """Emergency contact information"""
    name: str
    phone_number: str
    id: Optional[int] = None

@dataclass
class SOSMessage:
    """SOS message structure"""
    user_name: str
    coordinates: GPSCoordinates
    message: str
    timestamp: datetime

class EmergencyGestureDetector:
    """Detect emergency gestures from gesture recognition"""
    
    EMERGENCY_GESTURES = ["EMERGENCY", "SOS", "HELP", "ALERT"]
    
    @staticmethod
    def is_emergency_gesture(gesture_result: Dict) -> bool:
        """
        Check if detected gesture is an emergency gesture
        
        Args:
            gesture_result: Dictionary with gesture prediction
            
        Returns:
            bool: True if emergency gesture detected
        """
        gesture = gesture_result.get('gesture', '').upper()
        confidence = gesture_result.get('confidence', 0)
        
        return gesture in EmergencyGestureDetector.EMERGENCY_GESTURES and confidence > 0.8

class GPSLocationService:
    """Cross-platform GPS location fetching service"""
    
    @staticmethod
    def get_current_location() -> Optional[GPSCoordinates]:
        """
        Fetch current GPS coordinates across platforms
        
        Returns:
            GPSCoordinates or None if location unavailable
        """
        try:
            # Web platform - use browser geolocation API
            # This will be called from frontend JavaScript
            return None  # Location fetched via frontend
            
        except Exception as e:
            logger.error(f"GPS location fetch failed: {e}")
            return None

class SMSDispatcher:
    """Cross-platform SMS dispatch service"""
    
    def __init__(self, twilio_config: Dict[str, str] = None):
        """
        Initialize SMS dispatcher
        
        Args:
            twilio_config: Dictionary with Twilio credentials
        """
        self.twilio_config = twilio_config or {}
        self.client = None
        self._setup_twilio()
    
    def _setup_twilio(self):
        """Setup Twilio client"""
        try:
            account_sid = self.twilio_config.get('account_sid') or os.environ.get('TWILIO_ACCOUNT_SID')
            auth_token = self.twilio_config.get('auth_token') or os.environ.get('TWILIO_AUTH_TOKEN')
            
            if account_sid and auth_token:
                self.client = Client(account_sid, auth_token)
                logger.info("Twilio SMS dispatcher initialized")
            else:
                logger.warning("Twilio credentials not found")
                
        except Exception as e:
            logger.error(f"Failed to initialize Twilio: {e}")
    
    def send_sms(self, message: str, recipients: List[EmergencyContact]) -> Dict:
        """
        Send SMS to emergency contacts
        
        Args:
            message: SMS message content
            recipients: List of emergency contacts
            
        Returns:
            Dict with send results
        """
        results = {
            'successful': 0,
            'failed': 0,
            'details': []
        }
        
        if not self.client:
            logger.error("SMS service not configured")
            return results
        
        for contact in recipients:
            try:
                sms = self.client.messages.create(
                    body=message,
                    from_=self.twilio_config.get('phone_number') or os.environ.get('TWILIO_PHONE_NUMBER'),
                    to=contact.phone_number
                )
                
                results['successful'] += 1
                results['details'].append({
                    'contact': contact.name,
                    'phone': contact.phone_number,
                    'status': 'sent',
                    'message_id': sms.sid
                })
                
                logger.info(f"SMS sent to {contact.name}")
                
            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'contact': contact.name,
                    'phone': contact.phone_number,
                    'status': 'failed',
                    'error': str(e)
                })
                
                logger.error(f"Failed to send SMS to {contact.name}: {e}")
        
        return results

class SOSMessageGenerator:
    """Generate SOS messages with GPS coordinates"""
    
    @staticmethod
    def generate_message(user_name: str, coordinates: GPSCoordinates) -> str:
        """
        Generate SOS message with user name and GPS coordinates
        
        Args:
            user_name: Name of the user sending SOS
            coordinates: GPS coordinates
            
        Returns:
            Formatted SOS message
        """
        maps_url = f"https://maps.google.com/?q={coordinates.latitude},{coordinates.longitude}"
        
        message = f"""🚨 SOS! EMERGENCY 🚨
From: {user_name}
Location: {maps_url}
URGENT – Please respond immediately!"""
        
        return message

class EmergencySOSManager:
    """Main SOS management system"""
    
    def __init__(self, twilio_config: Dict[str, str] = None):
        """
        Initialize SOS manager
        
        Args:
            twilio_config: Twilio configuration dictionary
        """
        self.gesture_detector = EmergencyGestureDetector()
        self.gps_service = GPSLocationService()
        self.sms_dispatcher = SMSDispatcher(twilio_config)
        self.message_generator = SOSMessageGenerator()
        self.emergency_contacts = []
        self.user_name = "User"  # Default, will be updated
        
    def set_user_name(self, name: str):
        """Set the user name for SOS messages"""
        self.user_name = name
        
    def set_emergency_contacts(self, contacts: List[EmergencyContact]):
        """Set emergency contacts"""
        self.emergency_contacts = contacts
        
    def detect_emergency_gesture(self, gesture_result: Dict) -> bool:
        """
        Detect if gesture is emergency gesture
        
        Args:
            gesture_result: Gesture recognition result
            
        Returns:
            bool: True if emergency gesture detected
        """
        return self.gesture_detector.is_emergency_gesture(gesture_result)
    
    def trigger_sos(self, coordinates: GPSCoordinates) -> Dict:
        """
        Trigger SOS emergency alert
        
        Args:
            coordinates: Current GPS coordinates
            
        Returns:
            Dict with SOS execution results
        """
        if not self.emergency_contacts:
            return {'error': 'No emergency contacts configured'}
        
        # Generate SOS message
        message = self.message_generator.generate_message(
            user_name=self.user_name,
            coordinates=coordinates
        )
        
        # Send SMS to all contacts
        results = self.sms_dispatcher.send_sms(message, self.emergency_contacts)
        
        return {
            'message': 'SOS triggered successfully',
            'coordinates': {
                'latitude': coordinates.latitude,
                'longitude': coordinates.longitude
            },
            'contacts_notified': len(self.emergency_contacts),
            'sms_results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def trigger_sos_async(self, coordinates: GPSCoordinates) -> threading.Thread:
        """
        Trigger SOS asynchronously for background processing
        
        Args:
            coordinates: Current GPS coordinates
            
        Returns:
            Thread object for async processing
        """
        def sos_worker():
            return self.trigger_sos(coordinates)
        
        thread = threading.Thread(target=sos_worker)
        thread.start()
        return thread

# Global SOS manager instance
_sos_manager = None

def get_sos_manager(twilio_config: Dict[str, str] = None) -> EmergencySOSManager:
    """
    Get global SOS manager instance
    
    Args:
        twilio_config: Twilio configuration
        
    Returns:
        EmergencySOSManager instance
    """
    global _sos_manager
    if _sos_manager is None:
        _sos_manager = EmergencySOSManager(twilio_config)
    return _sos_manager

# Example usage
if __name__ == "__main__":
    # Initialize SOS manager
    sos_manager = get_sos_manager()
    
    # Set user name
    sos_manager.set_user_name("John Doe")
    
    # Set emergency contacts
    contacts = [
        EmergencyContact("Emergency Contact 1", "+1234567890"),
        EmergencyContact("Emergency Contact 2", "+1987654321")
    ]
    sos_manager.set_emergency_contacts(contacts)
    
    # Mock GPS coordinates
    coords = GPSCoordinates(37.7749, -122.4194)
    
    # Trigger SOS
    result = sos_manager.trigger_sos(coords)
    print(json.dumps(result, indent=2))
