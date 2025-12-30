# Technical Documentation - API Reference

## Overview

This document provides technical information about TechCorp's APIs and integration capabilities.

## Authentication

All API requests require authentication using one of the following methods:

### API Key Authentication
```
Authorization: Bearer YOUR_API_KEY
```

API keys can be generated in your account settings. Each key has configurable permissions and can be revoked at any time.

### OAuth 2.0
For user-specific operations, we support OAuth 2.0:
1. Redirect users to our authorization endpoint
2. Receive authorization code
3. Exchange code for access token
4. Use access token in API requests

Token endpoints:
- Authorization: `https://auth.techcorp.example.com/authorize`
- Token: `https://auth.techcorp.example.com/token`
- Refresh: `https://auth.techcorp.example.com/refresh`

## Rate Limits

API rate limits vary by plan:

| Plan | Requests/Minute | Requests/Day |
|------|-----------------|--------------|
| Basic | 60 | 10,000 |
| Professional | 300 | 100,000 |
| Enterprise | 1,000 | Unlimited |

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Timestamp when limit resets

## CloudSync Pro API

### Base URL
```
https://api.cloudsync.techcorp.example.com/v2
```

### Endpoints

#### List Files
```http
GET /files
```

Parameters:
- `path` (optional): Directory path, defaults to root
- `limit` (optional): Results per page, max 100
- `offset` (optional): Pagination offset

Response:
```json
{
  "files": [
    {
      "id": "file_abc123",
      "name": "document.pdf",
      "size": 1024000,
      "type": "file",
      "modified": "2024-01-15T10:30:00Z",
      "path": "/documents/document.pdf"
    }
  ],
  "total": 150,
  "limit": 50,
  "offset": 0
}
```

#### Upload File
```http
POST /files/upload
Content-Type: multipart/form-data
```

Parameters:
- `file` (required): File to upload
- `path` (optional): Destination path
- `overwrite` (optional): Replace existing file

Response:
```json
{
  "id": "file_xyz789",
  "name": "newfile.pdf",
  "size": 2048000,
  "upload_time": "2024-01-15T11:00:00Z"
}
```

#### Download File
```http
GET /files/{file_id}/download
```

Returns the file contents with appropriate Content-Type header.

#### Delete File
```http
DELETE /files/{file_id}
```

Response:
```json
{
  "success": true,
  "deleted_at": "2024-01-15T11:30:00Z"
}
```

### Share Files
```http
POST /files/{file_id}/share
```

Body:
```json
{
  "email": "recipient@example.com",
  "permissions": "view",
  "expires_at": "2024-02-15T00:00:00Z",
  "password_protected": true
}
```

## SecureAuth API

### Base URL
```
https://api.secureauth.techcorp.example.com/v1
```

### Verify User
```http
POST /auth/verify
```

Body:
```json
{
  "user_id": "user_123",
  "method": "totp",
  "code": "123456"
}
```

Response:
```json
{
  "verified": true,
  "user_id": "user_123",
  "timestamp": "2024-01-15T12:00:00Z"
}
```

### Send OTP
```http
POST /otp/send
```

Body:
```json
{
  "user_id": "user_123",
  "method": "sms",
  "phone": "+1234567890"
}
```

## DataFlow Analytics API

### Base URL
```
https://api.dataflow.techcorp.example.com/v1
```

### Execute Query
```http
POST /query
```

Body:
```json
{
  "datasource": "ds_abc123",
  "query": "SELECT * FROM sales WHERE date > '2024-01-01'",
  "format": "json"
}
```

### Get Dashboard Data
```http
GET /dashboards/{dashboard_id}/data
```

Parameters:
- `filters` (optional): JSON object with filter criteria
- `date_range` (optional): start and end dates

## Error Handling

All APIs return consistent error responses:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request was invalid",
    "details": {
      "field": "email",
      "reason": "Invalid email format"
    }
  },
  "request_id": "req_abc123"
}
```

### Common Error Codes
- `INVALID_REQUEST`: Request validation failed
- `UNAUTHORIZED`: Authentication required or failed
- `FORBIDDEN`: Insufficient permissions
- `NOT_FOUND`: Resource not found
- `RATE_LIMITED`: Too many requests
- `INTERNAL_ERROR`: Server error (retry recommended)

## Webhooks

You can configure webhooks to receive real-time notifications:

### Supported Events
- `file.created` - New file uploaded
- `file.deleted` - File deleted
- `file.shared` - File sharing updated
- `user.authenticated` - User login
- `user.mfa_failed` - Failed MFA attempt
- `report.generated` - Scheduled report ready

### Webhook Payload
```json
{
  "event": "file.created",
  "timestamp": "2024-01-15T12:00:00Z",
  "data": {
    "file_id": "file_abc123",
    "name": "document.pdf"
  },
  "signature": "sha256=abc123..."
}
```

### Signature Verification
All webhooks include an HMAC signature in the `X-TechCorp-Signature` header. Verify using your webhook secret:

```python
import hmac
import hashlib

def verify_signature(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

## SDKs

Official SDKs are available for:
- Python: `pip install techcorp-sdk`
- JavaScript/Node.js: `npm install @techcorp/sdk`
- Java: Maven Central
- Go: `go get github.com/techcorp/sdk-go`
- Ruby: `gem install techcorp`

## Support

For API support:
- Documentation: https://developers.techcorp.example.com
- API Status: https://status.techcorp.example.com
- Developer Forum: https://forum.techcorp.example.com
- Email: api-support@techcorp.example.com
