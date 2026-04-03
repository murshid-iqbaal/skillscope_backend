import json
import logging
import re
from typing import Optional, Dict, Any

from groq import AsyncGroq
from core.config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Exception Classes
# ──────────────────────────────────────────────

class GroqServiceError(Exception):
    def __init__(self, message: str, error_type: str = "groq_error"):
        self.message = message
        self.error_type = error_type
        super().__init__(message)

class GroqAuthError(GroqServiceError):
    def __init__(self):
        super().__init__("Invalid or missing Groq API key.", "auth_error")

class GroqRateLimitError(GroqServiceError):
    def __init__(self):
        super().__init__("Groq API rate limit reached.", "rate_limit_error")

class GroqTimeoutError(GroqServiceError):
    def __init__(self):
        super().__init__("Request to Groq API timed out.", "timeout_error")

class GroqNetworkError(GroqServiceError):
    def __init__(self):
        super().__init__("Network error connecting to Groq.", "network_error")

# ──────────────────────────────────────────────
# Internal Client Setup
# ──────────────────────────────────────────────

_client: Optional[AsyncGroq] = None

def get_client() -> AsyncGroq:
    global _client
    if _client is None:
        if not settings.GROQ_API_KEY:
            raise GroqAuthError()
        _client = AsyncGroq(
            api_key=settings.GROQ_API_KEY,
            timeout=settings.REQUEST_TIMEOUT,
        )
    return _client

def _handle_exception(e: Exception) -> None:
    error_str = str(e).lower()
    logger.error(f"Groq API error: {error_str}")
    
    if "rate_limit" in error_str:
        raise GroqRateLimitError()
    elif "timeout" in error_str:
        raise GroqTimeoutError()
    elif "authentication" in error_str or "api_key" in error_str:
        raise GroqAuthError()
    elif "connection" in error_str or "network" in error_str:
        raise GroqNetworkError()
    else:
        raise GroqServiceError(f"AI service error: {error_str}", "api_error")

# ──────────────────────────────────────────────
# Public AI Functions
# ──────────────────────────────────────────────

async def generate_chat_response(message: str) -> Dict[str, Any]:
    """
    Standalone function to generate a career-focused chat response.
    Standardized on llama3-70b-8192 for high-quality guidance.
    """
    client = get_client()
    try:
        logger.info("Sending chat request to Groq | model=llama3-70b-8192")
        completion = await client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": message}],
            temperature=0.7,
            max_tokens=1024,
        )
        
        response_text = completion.choices[0].message.content.strip()
        return {"response": response_text, "model": completion.model}

    except Exception as e:
        _handle_exception(e)

async def analyze_resume_ai(resume_text: str, job_role: str) -> Dict[str, Any]:
    """
    Standalone function to perform deep resume analysis using AI.
    Returns structured JSON results.
    """
    client = get_client()
    prompt = f"""You are an expert ATS resume analyzer.

Analyze the resume for the given job role.

Job Role: {job_role}

Resume:
{resume_text}

Return ONLY valid JSON:

{{
"matchScore": number (0-100),
"detectedSkills": [list of skills],
"missingSkills": [list of missing skills],
"suggestions": "short improvement suggestions"
}}

No explanation outside JSON."""

    try:
        logger.info(f"Sending resume analysis request to Groq | role={job_role} | model=llama3-70b-8192")
        completion = await client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a professional ATS analyzer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Strict deterministic output
            response_format={"type": "json_object"},
        )
        
        raw_content = completion.choices[0].message.content.strip()
        parsed_data = _parse_json_safely(raw_content)
        parsed_data["model"] = completion.model
        return parsed_data

    except Exception as e:
        _handle_exception(e)

def _parse_json_safely(raw_text: str) -> Dict[str, Any]:
    """Helper to safely extract and parse JSON from AI output."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Regex fallback for embedded JSON
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        logger.error(f"Failed to parse JSON: {raw_text}")
        raise GroqServiceError("Invalid JSON response from AI.", "parse_error")

async def health_check() -> Optional[str]:
    """Connectivity check for deployment validation."""
    try:
        await generate_chat_response("Say healthy")
        return None
    except Exception as e:
        return str(e)
