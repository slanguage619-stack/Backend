# SMS Not Sending - Troubleshooting Guide

## 🔍 Immediate SMS Fix Steps

### 1. **Twilio Configuration Issues**

**Problem**: SMS credentials not properly configured
**Solution**:
```bash
# Check current configuration
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('TWILIO_ACCOUNT_SID:', 'SET' if os.getenv('TWILIO_ACCOUNT_SID') else 'NOT SET')"

# Fix phone number format (add + prefix)
# Edit .env file:
TWILIO_PHONE_NUMBER=+18367755747  # Must start with +
```

### 2. **Restart Backend Server**

**Problem**: Backend not reading new .env configuration
**Solution**:
```bash
# Stop current backend
Ctrl+C in terminal running flask

# Restart with new configuration
python app.py
```

### 3. **Emergency Contacts Format**

**Problem**: Phone numbers not in international format
**Solution**:
```bash
# Check current contacts
python -c "
from app import EmergencyContact, db
contacts = EmergencyContact.query.all()
for c in contacts:
    print(f'{c.name}: {c.phone_number}')
"

# Ensure all contacts have +country_code format
# Example: +15551234567 (US), +919876543210 (India)
```

### 4. **Test SMS Directly**

**Test 1: Backend API**
```bash
curl -X POST http://localhost:5001/api/sos \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Test 2: Twilio Connection**
```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
from twilio.rest import Client

client = Client(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN'))
account = client.api.accounts(os.getenv('TWILIO_ACCOUNT_SID')).fetch()
print('✅ Twilio connected:', account.friendly_name)
"
```

### 5. **Common Issues & Fixes**

| Issue | Fix |
|-------|-----|
| **Phone number format** | Add + prefix: +1234567890 |
| **Trial account** | Upgrade Twilio account or verify numbers |
| **Backend restart** | Restart flask after .env changes |
| **Contact format** | Ensure +country_code format |
| **Network issues** | Check internet connectivity |

### 6. **Quick Verification Script**

```bash
# Run this diagnostic
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

# Check all requirements
required = ['TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'TWILIO_PHONE_NUMBER']
missing = [r for r in required if not os.getenv(r)]

if missing:
    print('❌ Missing:', missing)
else:
    print('✅ All credentials configured')
    
    # Test phone format
    phone = os.getenv('TWILIO_PHONE_NUMBER')
    if phone and not phone.startswith('+'):
        print('❌ Phone needs + prefix')
    else:
        print('✅ Phone format valid')
"
```

### 7. **Immediate Action Steps**

1. **Fix phone number format**:
   ```bash
   # Edit .env file
   TWILIO_PHONE_NUMBER=+18367755747  # Add + prefix
   ```

2. **Restart backend**:
   ```bash
   # Stop current server
   Ctrl+C
   
   # Restart
   python app.py
   ```

3. **Test SMS**:
   ```bash
   # Test with curl
   curl -X POST http://localhost:5001/api/sos \
     -H "Content-Type: application/json" \
     -d '{"test": true}'
   ```

4. **Check emergency contacts**:
   - Ensure all contacts have +country_code format
   - Verify phone numbers are valid

### 8. **Debug Mode**

```bash
# Run backend with debug logging
python app.py

# Check console for SMS errors
# Look for: "SMS sent successfully" or error messages
```

## ✅ SMS Should Now Work

After fixing:
- Phone number format (+ prefix)
- Backend restart
- Valid emergency contacts
- Twilio credentials loaded

SMS delivery should work immediately!
