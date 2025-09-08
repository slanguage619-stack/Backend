#!/usr/bin/env python3
"""
Twilio Setup Script for SignCrypt SOS System
Helps configure Twilio credentials for SMS delivery
"""

import os
import sys
import json
from pathlib import Path

def setup_twilio():
    """Interactive Twilio setup script"""
    print("🚨 SignCrypt SOS System - Twilio Setup")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        env_file.touch()
    
    print("\n📱 To set up SMS delivery, you need:")
    print("1. Twilio Account SID")
    print("2. Twilio Auth Token") 
    print("3. Twilio Phone Number")
    print("\nGet these from: https://console.twilio.com/")
    
    # Get user input
    account_sid = input("\nEnter Twilio Account SID: ").strip()
    auth_token = input("Enter Twilio Auth Token: ").strip()
    phone_number = input("Enter Twilio Phone Number (+1234567890): ").strip()
    
    # Validate inputs
    if not all([account_sid, auth_token, phone_number]):
        print("❌ All fields are required!")
        return False
    
    if not phone_number.startswith('+'):
        phone_number = '+' + phone_number
    
    # Create .env file
    with open('.env', 'w') as f:
        f.write(f"TWILIO_ACCOUNT_SID={account_sid}\n")
        f.write(f"TWILIO_AUTH_TOKEN={auth_token}\n")
        f.write(f"TWILIO_PHONE_NUMBER={phone_number}\n")
    
    # Create setup confirmation
    setup_data = {
        "account_sid": "***" + account_sid[-4:],
        "phone_number": phone_number,
        "setup_date": "2025-08-11",
        "status": "configured"
    }
    
    with open('twilio_setup.json', 'w') as f:
        json.dump(setup_data, f, indent=2)
    
    print("\n✅ Twilio configuration saved!")
    print("📁 Files created:")
    print("  - .env (with credentials)")
    print("  - twilio_setup.json (confirmation)")
    
    print("\n🚀 To apply changes:")
    print("1. Restart the backend server")
    print("2. Test SMS delivery")
    
    return True

def test_twilio_connection():
    """Test Twilio connection"""
    try:
        from twilio.rest import Client
        
        # Load from environment
        account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
        auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
        
        if not all([account_sid, auth_token]):
            print("❌ Twilio credentials not found in environment")
            return False
            
        client = Client(account_sid, auth_token)
        
        # Test account info
        account = client.api.accounts(account_sid).fetch()
        print(f"✅ Twilio connection successful!")
        print(f"   Account: {account.friendly_name}")
        print(f"   Status: {account.status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Twilio connection failed: {e}")
        return False

def validate_phone_numbers():
    """Validate emergency contacts phone numbers"""
    print("\n📞 Validating emergency contacts...")
    
    # This would check the database for valid phone numbers
    # For now, provide guidance
    print("Ensure emergency contacts have:")
    print("- International format (+1234567890)")
    print("- Valid country codes")
    print("- No spaces or special characters")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_twilio_connection()
    else:
        setup_twilio()
        validate_phone_numbers()
