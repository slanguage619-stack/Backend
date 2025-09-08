"""
Chatbot API Integration
Integrates with the Railway chatbot API at https://chatapi-production-42f0.up.railway.app/chat
"""

import requests
import json
import logging

class ChatbotAPI:
    """Chatbot API client for Railway chatbot"""
    
    def __init__(self, base_url=None, timeout=30):
        """
        Initialize Chatbot API
        
        Args:
            base_url (str): Base URL for the chatbot API
            timeout (int): Request timeout in seconds
        """
        self.base_url = base_url or "https://chatapi-production-e44c.up.railway.app"
        # Try different possible endpoints
        self.chat_endpoints = [
            f"{self.base_url}/chat",
            f"{self.base_url}/api/chat",
            f"{self.base_url}/chatbot"
        ]
        self.timeout = timeout
        
    def send_message(self, message, user_id=None, context=None):
        """
        Send message to chatbot API
        
        Args:
            message (str): User message
            user_id (str): Optional user identifier
            context (dict): Optional context data
            
        Returns:
            dict: Chatbot response
        """
        try:
            payload = {
                "message": message,
                "user_id": user_id or "default_user",
                "context": context or {}
            }
            
            # Handle file content in context
            file_content = context.get('file_content') if context else None
            file_name = context.get('file_name') if context else None
            file_type = context.get('file_type') if context else None
            
            if file_content and file_name:
                # Enhance the message with file information
                enhanced_message = f"{message}\n\n[File uploaded: {file_name} ({file_type})]"
                
                # For text files, include the content
                if file_type and file_type.startswith('text/'):
                    enhanced_message += f"\n\nFile content:\n{file_content[:1000]}..." if len(file_content) > 1000 else f"\n\nFile content:\n{file_content}"
                elif file_type and file_type.startswith('image/'):
                    enhanced_message += "\n\n[Image file uploaded - processing image content]"
                else:
                    enhanced_message += f"\n\n[File type: {file_type}]"
                
                payload['message'] = enhanced_message
            
            # Try multiple endpoints
            last_error = None
            for endpoint in self.chat_endpoints:
                try:
                    response = requests.post(
                        endpoint,
                        json=payload,
                        timeout=self.timeout,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        last_error = {
                            "endpoint": endpoint,
                            "status_code": response.status_code,
                            "response": response.text
                        }
                        logging.warning(f"Chatbot endpoint {endpoint} returned {response.status_code}: {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    last_error = {
                        "endpoint": endpoint,
                        "error": str(e)
                    }
                    logging.warning(f"Chatbot endpoint {endpoint} failed: {e}")
                    continue
            
            # All endpoints failed
            error_details = f"All chatbot endpoints failed. Last error: {last_error}"
            logging.error(f"Chatbot API error: {error_details}")
            
            # Provide contextual fallback for file uploads
            if file_content and file_name:
                return {
                    "error": "Service unavailable",
                    "message": f"I've received your file '{file_name}'. While the external chatbot service is temporarily unavailable, I can help you with file analysis once the service is restored.",
                    "details": error_details,
                    "suggestion": "The chatbot service may be temporarily down. Please wait a few minutes and try again."
                }
            
            return {
                "error": "Service unavailable",
                "message": "Chatbot service is currently unavailable. Please try again later.",
                "details": error_details,
                "suggestion": "The chatbot service may be temporarily down. Please wait a few minutes and try again."
            }
                
        except Exception as e:
            logging.error(f"Unexpected chatbot error: {e}")
            return {
                "error": str(e),
                "message": "An unexpected error occurred. Please try again.",
                "details": str(e)
            }
    
    def get_chat_response(self, user_input, user_context=None):
        """
        Get chatbot response for user input
        
        Args:
            user_input (str): User's message
            user_context (dict): Additional context
            
        Returns:
            str: Chatbot response text
        """
        response = self.send_message(user_input, context=user_context)
        
        if "error" in response:
            return response.get("message", "Service unavailable")
        
        return response.get("response", response.get("message", "No response"))

# Global chatbot instance
chatbot = ChatbotAPI()

def get_chatbot_response(user_input, user_id=None, context=None):
    """Convenience function to get chatbot response"""
    return chatbot.get_chat_response(user_input, user_context=context)

def send_chatbot_message(message, user_id=None, context=None):
    """Convenience function to send message to chatbot"""
    return chatbot.send_message(message, user_id=user_id, context=context)
