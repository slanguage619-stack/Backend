# Indian Phone Number Format for SMS

## 📱 Correct Indian Phone Number Format

### ✅ **Indian Mobile Numbers**

**Format**: `+91` followed by 10-digit mobile number

**Examples**:
- `+919876543210` (correct)
- `+918367755747` (correct)
- `8367755747` (incorrect - missing +91)
- `918367755747` (incorrect - missing +)

### 🔧 **How to Convert Indian Numbers**

**From**: 8367755747
**To**: `+918367755747`

**Steps**:
1. Add `+` prefix
2. Add country code `91` for India
3. Keep the 10-digit mobile number

### 📋 **Emergency Contact Examples**

```javascript
// Correct Indian format
{
  name: "Emergency Contact",
  phone_number: "+919876543210"
}

// Multiple Indian contacts
[
  { name: "Mom", phone_number: "+919876543210" },
  { name: "Dad", phone_number: "+919112233445" },
  { name: "Friend", phone_number: "+9195566778899" }
]
```

### ⚡ **Quick Fix Commands**

```bash
# Update existing contacts to Indian format
python -c "
from app import EmergencyContact, db
contacts = EmergencyContact.query.all()
for contact in contacts:
    # Convert to Indian format
    if not contact.phone_number.startswith('+'):
        contact.phone_number = '+91' + contact.phone_number
        print(f'Updated {contact.name}: {contact.phone_number}')
db.session.commit()
"
```

### 🔍 **Validation Script**

```bash
# Check current contact formats
python -c "
from app import EmergencyContact, db
contacts = EmergencyContact.query.all()
print('Current contacts:')
for contact in contacts:
    is_valid = contact.phone_number.startswith('+')
    print(f'  {contact.name}: {contact.phone_number} {"✅" if is_valid else "❌"}')
"
```

### 📞 **Common Indian Mobile Prefixes**

| Prefix | Example | Complete |
|--------|---------|----------|
| 9xxxxxxxxx | 9876543210 | +919876543210 |
| 8xxxxxxxxx | 8765432109 | +918765432109 |
| 7xxxxxxxxx | 7654321098 | +917654321098 |
| 6xxxxxxxxx | 6543210987 | +916543210987 |

### ✅ **SMS Test with Indian Numbers**

```bash
# Test SMS to Indian number
curl -X POST http://localhost:5001/api/sos \
  -H "Content-Type: application/json" \
  -d '{
    "coordinates": {"latitude": 19.0760, "longitude": 72.8777},
    "userName": "Test User",
    "message": "🚨 SOS! EMERGENCY 🚨\nFrom: Test User\nLocation: https://maps.google.com/?q=19.0760,72.8777\nURGENT – Please respond immediately!"
  }'
```

### 🚨 **Immediate Fix**

1. **Update phone numbers to Indian format**:
   ```bash
   # Add +91 prefix to all Indian numbers
   # Example: 8367755747 → +918367755747
   ```

2. **Verify format**:
   ```bash
   python -c "
   from app import EmergencyContact, db
   contacts = EmergencyContact.query.all()
   for c in contacts:
       print(f'{c.name}: {c.phone_number}')
   ```

3. **Test SMS delivery**:
   ```bash
   # After backend restart, test with Indian numbers
   curl -X POST http://localhost:5001/api/sos
   ```

### 📋 **Complete Indian Number Setup**

```javascript
// Frontend - Add Indian contact
const indianContact = {
  name: "Emergency Contact",
  phone_number: "+919876543210"  // Indian format
};

// Backend - Ensure Indian format
# All emergency contacts must be +91XXXXXXXXXX
```

**Key**: Every Indian mobile number needs `+91` prefix for SMS delivery to work!
