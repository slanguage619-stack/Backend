# Twilio SMS Setup Guide for SignCrypt SOS

## 🔧 SMS Integration Setup

The SOS feature now supports real SMS delivery via Twilio. Follow these steps to enable SMS notifications:

### 1. Create Twilio Account
- Go to [twilio.com](https://www.twilio.com)
- Sign up for a free trial account
- Verify your phone number for testing

### 2. Get Your Credentials
- **Account SID**: Found in your Twilio Console dashboard
- **Auth Token**: Found in your Twilio Console dashboard
- **Phone Number**: Get a Twilio phone number from the Console

### 3. Environment Variables Setup

Create a `.env` file in your backend directory:

```bash
# .env file
TWILIO_ACCOUNT_SID=your_actual_account_sid_here
TWILIO_AUTH_TOKEN=your_actual_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890  # Your Twilio phone number
```

### 4. Install Dependencies

The Twilio Python SDK is already included in requirements.txt. If not installed:

```bash
pip install twilio python-dotenv
```

### 5. Update Environment Variables

In your terminal or deployment environment:

```bash
# Linux/Mac
export TWILIO_ACCOUNT_SID="your_actual_account_sid"
export TWILIO_AUTH_TOKEN="your_actual_auth_token"
export TWILIO_PHONE_NUMBER="+1234567890"

# Windows
set TWILIO_ACCOUNT_SID=your_actual_account_sid
set TWILIO_AUTH_TOKEN=your_actual_auth_token
set TWILIO_PHONE_NUMBER=+1234567890
```

### 6. Test SMS Functionality

1. **Add Emergency Contacts**: Use the frontend to add contacts with valid phone numbers
2. **Send Test SOS**: Click the SOS button to trigger SMS
3. **Check Logs**: Monitor console for SMS delivery confirmations

### 7. Phone Number Format

Ensure emergency contacts are saved with proper international format:
- **US Numbers**: +1 followed by 10 digits (e.g., +15551234567)
- **Other Countries**: Use proper country codes

### 8. Troubleshooting

**SMS Not Delivering?**
- Check Twilio Console for error logs
- Verify phone numbers are in correct format
- Ensure Twilio account has sufficient balance
- Check environment variables are properly set

**Error Messages:**
- "SMS service not configured": Environment variables not set
- "Failed to send SMS": Invalid phone number or Twilio issue
- "Twilio not configured": Missing credentials

### 9. Production Deployment

For production:
1. Upgrade Twilio account from trial
2. Verify sender phone numbers
3. Set up proper error handling and monitoring
4. Consider rate limiting for SMS

### 10. Testing Numbers

For testing, you can use:
- Your own phone number (must be verified with Twilio)
- Twilio's test numbers
- Other verified numbers in your Twilio account

## 🚨 Emergency Message Format

The SMS will contain:
```
🚨 EMERGENCY SOS ALERT 🚨

This is an emergency alert from SignCrypt.
Please check on the sender immediately.

Sent via SignCrypt Emergency System
```

## 📞 Support

- **Twilio Support**: https://support.twilio.com
- **SignCrypt Issues**: Check console logs for detailed error messages
