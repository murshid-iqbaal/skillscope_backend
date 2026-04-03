# SkillScope AI — Backend

Production-ready FastAPI backend for the SkillScope AI mobile app chatbot, powered by **Groq** for ultra-fast LLM inference.

---

## Features

- `POST /chat` — AI career & tech assistant powered by Groq
- `GET /health` — Liveness probe for Render / load balancers
- `GET /health/groq` — Deep connectivity check against Groq API
- Full CORS support for Flutter mobile clients
- Typed error handling (auth, rate limit, timeout, network)
- Auto-generated Swagger docs at `/docs`
- One-command deploy to Render

---

## Project Structure

```
backend/
├── core/
│   └── config.py           # Env var loading via python-dotenv
├── models/
│   └── chat_models.py      # Pydantic request/response models
├── routers/
│   └── chat.py             # POST /chat endpoint
├── services/
│   └── groq_service.py     # Groq API integration (async httpx)
├── utils/
│   └── prompt_builder.py   # System prompt + message builder
├── main.py                 # FastAPI app, CORS, lifespan hooks
├── requirements.txt
├── render.yaml             # Render deployment config
└── .env.example
```

---

## Quick Start (Local)

### 1. Clone & enter the directory
```bash
cd backend
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get a free Groq API key at: https://console.groq.com/

### 5. Run the server
```bash
uvicorn main:app --reload --port 8000
```

API will be live at: `http://localhost:8000`
Swagger UI at: `http://localhost:8000/docs`

---

## API Reference

### `POST /chat`

Send a message to SkillScope AI.

**Request**
```json
{
  "message": "What skills are needed for a Flutter developer?",
  "conversation_id": "optional-session-id"
}
```

**Response `200`**
```json
{
  "response": "To become a Flutter developer, you should learn:\n\n• Dart programming language...",
  "model": "llama3-70b-8192",
  "conversation_id": "optional-session-id"
}
```

**Error responses**

| Status | Meaning                        |
|--------|--------------------------------|
| 422    | Validation error (empty msg)   |
| 401    | Invalid Groq API key           |
| 429    | Groq rate limit exceeded       |
| 408    | Request timeout                |
| 503    | Groq API unreachable           |
| 502    | Unexpected Groq API error      |
| 500    | Internal server error          |

---

### `GET /health`
Returns `200` if the server is running. Used by Render for health checks.

### `GET /health/groq`
Tests live connectivity to the Groq API. Returns `503` if Groq is unreachable.

---

## Flutter Integration

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

Future<String> askSkillScope(String message) async {
  final response = await http.post(
    Uri.parse('https://YOUR-RENDER-URL.onrender.com/chat'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'message': message}),
  );

  if (response.statusCode == 200) {
    final data = jsonDecode(response.body);
    return data['response'] as String;
  } else {
    throw Exception('Chat error: ${response.statusCode}');
  }
}
```

---

## Deploying to Render

### Option A — Automatic (render.yaml)

1. Push this folder to a GitHub repository
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Apply**
5. In the Render dashboard → **Environment** → add:
   ```
   GROQ_API_KEY = your_actual_key_here
   ```
6. Deploy — your API will be live at `https://skillscope-ai-backend.onrender.com`

### Option B — Manual Setup

| Setting         | Value                                              |
|-----------------|----------------------------------------------------|
| Environment     | Python                                             |
| Build Command   | `pip install -r requirements.txt`                  |
| Start Command   | `uvicorn main:app --host 0.0.0.0 --port 10000`     |
| Health Check    | `/health`                                          |

---

## Available Groq Models

Set via `GROQ_MODEL` environment variable:

| Model                  | Context  | Best For                     |
|------------------------|----------|------------------------------|
| `llama3-70b-8192`      | 8K       | Best quality (default)       |
| `llama3-8b-8192`       | 8K       | Faster, lighter              |
| `mixtral-8x7b-32768`   | 32K      | Long context conversations   |
| `gemma-7b-it`          | 8K       | Efficient alternative        |

---

## Environment Variables

| Variable          | Required | Default            | Description             |
|-------------------|----------|--------------------|-------------------------|
| `GROQ_API_KEY`    | ✅ Yes   | —                  | Your Groq API key       |
| `GROQ_MODEL`      | No       | `llama3-70b-8192`  | Groq model to use       |
| `REQUEST_TIMEOUT` | No       | `30`               | HTTP timeout in seconds |
