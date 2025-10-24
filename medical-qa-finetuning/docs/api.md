# API Documentation

## Endpoints

### POST /generate
Generate medical response

**Request:**
```json
{
  "question": "string",
  "temperature": 0.7,
  "max_length": 100
}
```

**Response:**
```json
{
  "response": "string",
  "safety_report": {},
  "inference_time_ms": 87
}
```
