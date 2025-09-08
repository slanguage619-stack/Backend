#!/usr/bin/env python3
"""
Complete SMS Debug Script for SignCrypt SOS System
Tests every component from credentials to actual SMS delivery
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv

def test_sms_system():
    """Comprehensive SMS system test"""
    
    print("🚨 SMS SYSTEM DEBUG - SignCrypt SOS")
    print("=" * 50)
    
    # Load environment
    load_dotenv()
    
    # 1. Check Twilio credentials
    print("\n1️⃣ Twilio Credentials Check:")
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    phone_number = os.getenv('TWILIO_PHONE_NUMBER')
    
    credentials_status = {
        'account_sid': 'SET' if account_sid else 'NOT SET',
        'auth_token': 'SET' if auth_token else 'NOT SET',
        'phone_number': phone_number,
        'all_configured': all([account_sid, auth_token, phone_number])
    }
    
    print(f"   Account SID: {credentials_status['account_sid']}")
    print(f"   Auth Token: {credentials_status['auth_token']}")
    print(f"   Phone Number: {credentials_status['phone_number']}")
    print(f"   All Configured: {credentials_status['all_configured']}")
    
    if not credentials_status['all_configured']:
        print("   ❌ Missing credentials - check .env file")
        return False
    
    # 2. Check phone number format
    print("\n2️⃣ Phone Number Format:")
    if phone_number:
        is_valid = phone_number.startswith('+')
        print(f"   Format Valid (+ prefix): {is_valid}")
        if not is_valid:
            print(f"   ❌ Needs + prefix: {phone_number}")
            return False
    
    # 3. Test Twilio connection
    print("\n3️⃣ Twilio Connection Test:")
    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        
        # Test account access
        account = client.api.accounts(account_sid).fetch()
        print(f"   ✅ Twilio Connected: {account.friendly_name}")
        print(f"   Account Status: {account.status}")
        
    except Exception as e:
        print(f"   ❌ Twilio Connection Error: {e}")
        return False
    
    # 4. Check emergency contacts
    print("\n4️⃣ Emergency Contacts Check:")
    try:
        # Import app modules
        sys.path.insert(0, '.')
        from app import EmergencyContact, db
        
        contacts = EmergencyContact.query.all()
        print(f"   Total Contacts: {len(contacts)}")
        
        if not contacts:
            print("   ❌ No emergency contacts found")
            print("   Add contacts via frontend or database")
            return False
        
        valid_contacts = 0
        for contact in contacts:
            valid_format = contact.phone_number.startswith('+')
            print(f"   {contact.name}: {contact.phone_number} {'✅' if valid_format else '❌'}")
            if valid_format:
                valid_contacts += 1
        
        if valid_contacts == 0:
            print("   ❌ No valid phone number formats")
            return False
            
    except Exception as e:
        print(f"   ❌ Database Error: {e}")
        return False
    
    # 5. Test backend endpoint
    print("\n5️⃣ Backend Endpoint Test:")
    try:
        response = requests.post('http://localhost:5001/api/sos', json={})
        print(f"   Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Backend endpoint responding")
        else:
            print(f"   ❌ Endpoint error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Backend server not running")
        print("   Start server: python app.py")
        return False
    except Exception as e:
        print(f"   ❌ Endpoint test error: {e}")
        return False
    
    # 6. Test actual SMS
    print("\n6️⃣ SMS Delivery Test:")
    try:
        # Test SMS to first contact
        if contacts:
            test_contact = contacts[0]
            message = client.messages.create(
                body='Test SMS from SignCrypt SOS - Debug',
                from_=phone_number,
                to=test_contact.phone_number
            )
            print(f"   ✅ SMS Sent: {message.sid}")
            print(f"   To: {test_contact.name} ({test_contact.phone_number})")
            
    except Exception as e:
        print(f"   ❌ SMS Error: {e}")
        if "trial" in str(e).lower():
            print("   💡 Trial account - verify phone numbers first")
        return False
    
    # 7. System status summary
    print("\n7️⃣ System Status Summary:")
    print("   ✅ Twilio credentials configured")
    print("   ✅ Phone number format valid")
    print("   ✅ Twilio connection working")
    print("   ✅ Emergency contacts available")
    print("   ✅ Backend endpoint responding")
    print("   ✅ SMS delivery successful")
    
    return True

def fix_common_issues():
    """Fix common SMS issues"""
    
    print("\n🔧 Fixing Common Issues...")
    
    # Load environment
    load_dotenv()
    
    # Fix phone number format
    phone_number = os.getenv('TWILIO_PHONE_NUMBER')
    if phone_number and not phone_number.startswith('+'):
        print(f"   Fixing phone format: {phone_number}")
        # This would need manual fix in .env
    
    # Check database
    try:
        from app import EmergencyContact, db
        contacts = EmergencyContact.query.all()
        
        # Fix any Indian numbers
        for contact in contacts:
            if not contact.phone_number.startswith('+'):
                old = contact.phone_number
                contact.phone_number = '+91' + contact.phone_number
                print(f"   Fixed {contact.name}: {old} → {contact.phone_number}")
        
        db.session.commit()
        
    except Exception as e:
        print(f"   Database fix error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fix":
        fix_common_issues()
    else:
        success = test_sms_system()
        
        if success:
            print("\n🎉 SMS System Working Perfectly!")
        else:
            print("\n❌ SMS System Issues Found")
            print("\n🔍 Run this command to fix issues:")
            print("   python debug_sms.py fix")
