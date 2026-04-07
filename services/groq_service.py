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
        super().__init__("Invalid or missing Groq API key. Please check your GROQ_API_KEY.", "auth_error")

class GroqRateLimitError(GroqServiceError):
    def __init__(self):
        super().__init__("Groq API rate limit reached. Please try again later.", "rate_limit_error")

class GroqTimeoutError(GroqServiceError):
    def __init__(self):
        super().__init__("Request to Groq API timed out. Try reducing resume length.", "timeout_error")

class GroqNetworkError(GroqServiceError):
    def __init__(self):
        super().__init__("Network error connecting to Groq. Please check your connection.", "network_error")

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
    logger.error(f"Groq API error encountered: {error_str}")
    
    if "rate_limit" in error_str:
        raise GroqRateLimitError()
    elif "timeout" in error_str:
        raise GroqTimeoutError()
    elif "authentication" in error_str or "api_key" in error_str:
        raise GroqAuthError()
    elif "connection" in error_str or "network" in error_str:
        raise GroqNetworkError()
    else:
        raise GroqServiceError(f"Unexpected AI service error: {error_str}", "api_error")

# ──────────────────────────────────────────────
# Public AI Functions
# ──────────────────────────────────────────────

async def generate_chat_response(message: str) -> Dict[str, Any]:
    """
    Generate a career-focused chat response.
    Standardized on llama-3.1-8b-instant for fast, low-latency responses.
    """
    client = get_client()
    try:
        logger.info(f"Chat request to Groq | model=llama-3.1-8b-instant")
        completion = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are SkillScope AI, a career and technology mentor. Give concise, helpful advice."},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        
        response_text = completion.choices[0].message.content.strip()
        # Standardized on 'reply' key as per newest request
        return {"reply": response_text, "model": completion.model}

    except Exception as e:
        _handle_exception(e)

async def generate_skill_resources(skill_name: str) -> Dict[str, Any]:
    """
    Generate high-quality learning resources for a specific skill using Groq.
    This generates 3-5 direct resource links for the given skill.
    """
    client = get_client()
    prompt = f"""You are a master technical educator and career mentor.
    
    Task: Provide 3-5 high-quality, direct learning resources for the skill: {skill_name}.
    
    Return ONLY a perfectly formatted JSON object with this exact structure:
    {{
      "skill": "{skill_name}",
      "recommendedResources": [
        {{
          "title": "Resource title (e.g. Flutter State Management Masterclass)",
          "url": "VALID FULL HTTPS URL — must start with https://",
          "description": "One sentence about why this resource is excellent for learning {skill_name}",
          "platform": "YouTube / Coursera / Udemy / Official Docs"
        }}
      ]
    }}
    
    STRICT URL RULES:
    - Every url MUST start with https://
    - Focus on primary documentation, reputable YouTube channels, and top courses.
    - No broken links or generic search pages.
    - DO NOT include null or placeholder URLs.
    
    No conversational text outside the JSON block."""

    try:
        logger.info(f"Skill resources request to Groq | skill={skill_name} | model=llama-3.1-8b-instant")
        completion = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a professional educational consultant. Always return valid JSON with direct, functional HTTPS URLs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw_content = completion.choices[0].message.content.strip()
        parsed_data = _parse_json_safely(raw_content)
        
        # Sanitize and ensure structure matches
        parsed_data = _sanitize_resources(parsed_data)
        return parsed_data

    except Exception as e:
        _handle_exception(e)

async def analyze_resume_ai(resume_text: str, job_role: str) -> Dict[str, Any]:
    """
    Analyze a resume against a job role and return structured analysis + resources.
    Standardized on llama-3.1-8b-instant.
    """
    client = get_client()
    prompt = f"""You are an expert career assistant and ATS (Applicant Tracking System) analyzer.

Task: Analyze the resume for the job role: {job_role}.

Resume Text:
{resume_text}

Return ONLY a perfectly formatted JSON object with this exact structure:
{{
  "matchScore": number (0-100),
  "detectedSkills": ["skill1", "skill2"],
  "missingSkills": ["missing1", "missing2"],
  "recommendedResources": [
    {{
      "title": "Resource title (e.g. Flutter State Management Masterclass)",
      "url": "VALID FULL HTTPS URL — must start with https://",
      "description": "Short description of why this resource helps",
      "platform": "YouTube / Coursera / Udemy / Official Docs"
    }}
  ]
}}

STRICT URL RULES — violations will cause system failure:
- Every url MUST start with https://
- For YouTube videos: use full watch link → https://www.youtube.com/watch?v=VIDEO_ID
- For Coursera: https://www.coursera.org/learn/COURSE-SLUG
- For Udemy: https://www.udemy.com/course/COURSE-SLUG/
- For official docs: use the real documentation URL
- DO NOT return plain text like "search on YouTube for Flutter tutorial"
- DO NOT return https://www.youtube.com alone (homepage — invalid)
- DO NOT return https://www.google.com alone (homepage — invalid)
- DO NOT include null, empty, or incomplete URLs

No conversational text outside the JSON block."""

    try:
        logger.info(f"Resume analysis to Groq | role={job_role} | model=llama-3.1-8b-instant")
        completion = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a professional resume auditor and career coach. Always return valid JSON with real, direct HTTPS URLs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Keep it strictly deterministic
            response_format={"type": "json_object"},
        )

        raw_content = completion.choices[0].message.content.strip()
        parsed_data = _parse_json_safely(raw_content)
        parsed_data["model"] = completion.model

        # Sanitize all resource URLs before returning — no invalid URL leaves the backend
        parsed_data = _sanitize_resources(parsed_data)
        return parsed_data

    except Exception as e:
        _handle_exception(e)

def _sanitize_resources(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and filter resources to ensure URLs are strictly HTTPS."""
    if "recommendedResources" in data and isinstance(data["recommendedResources"], list):
        valid_resources = []
        for res in data["recommendedResources"]:
            url = res.get("url", "")
            if isinstance(url, str) and url.startswith("https://") and len(url) > 12:
                valid_resources.append(res)
        data["recommendedResources"] = valid_resources
    return data

def _parse_json_safely(raw_text: str) -> Dict[str, Any]:
    """Extract and parse JSON from AI string with regex fallback."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        logger.error(f"JSON Parsing failed for AI output: {raw_text}")
        raise GroqServiceError("The AI service returned an unreadable response format.", "parse_error")

async def health_check() -> Optional[str]:
    """Diagnostic health check for deployment monitors."""
    try:
        await generate_chat_response("ping")
        return None
    except Exception as e:
        return str(e)
