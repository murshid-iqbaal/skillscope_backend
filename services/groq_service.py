import json
import logging
import re
from typing import Any, Dict, List, Optional

from groq import AsyncGroq

from core.config import settings

logger = logging.getLogger(__name__)


class GroqServiceError(Exception):
    def __init__(self, message: str, error_type: str = "groq_error"):
        self.message = message
        self.error_type = error_type
        super().__init__(message)


class GroqAuthError(GroqServiceError):
    def __init__(self):
        super().__init__(
            "Invalid or missing Groq API key. Please check your GROQ_API_KEY.",
            "auth_error",
        )


class GroqRateLimitError(GroqServiceError):
    def __init__(self):
        super().__init__(
            "Groq API rate limit reached. Please try again later.",
            "rate_limit_error",
        )


class GroqTimeoutError(GroqServiceError):
    def __init__(self):
        super().__init__(
            "Request to Groq API timed out. Try reducing resume length.",
            "timeout_error",
        )


class GroqNetworkError(GroqServiceError):
    def __init__(self):
        super().__init__(
            "Network error connecting to Groq. Please check your connection.",
            "network_error",
        )


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


def _handle_exception(error: Exception) -> None:
    error_str = str(error).lower()
    logger.error("Groq API error encountered: %s", error_str)

    if "rate_limit" in error_str:
        raise GroqRateLimitError()
    if "timeout" in error_str:
        raise GroqTimeoutError()
    if "authentication" in error_str or "api_key" in error_str:
        raise GroqAuthError()
    if "connection" in error_str or "network" in error_str:
        raise GroqNetworkError()
    raise GroqServiceError(f"Unexpected AI service error: {error_str}", "api_error")


async def generate_chat_response(message: str) -> Dict[str, Any]:
    """
    Generate a career-focused chat response.
    """
    client = get_client()
    try:
        logger.info("Chat request to Groq | model=llama-3.1-8b-instant")
        completion = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are SkillScope AI, a career and technology mentor. "
                        "Give concise, helpful advice."
                    ),
                },
                {"role": "user", "content": message},
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        response_text = completion.choices[0].message.content.strip()
        return {"reply": response_text, "model": completion.model}

    except Exception as error:
        _handle_exception(error)


async def generate_skill_resources(skill_name: str) -> Dict[str, Any]:
    """
    Generate high-quality learning resources for a specific skill using Groq.
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
      "url": "VALID FULL HTTPS URL and must start with https://",
      "description": "One sentence about why this resource is excellent for learning {skill_name}",
      "platform": "YouTube / Coursera / Udemy / Official Docs"
    }}
  ]
}}

STRICT URL RULES:
1. Every URL MUST start with https://
2. TRUSTED DOMAINS: Prefer official docs (e.g., flutter.dev, dart.dev, react.dev, python.org, aws.amazon.com), YouTube.com, Coursera.org, or Udemy.com.
3. NO HALLUCINATIONS: Do NOT guess specific sub-page paths or video IDs if you are not 100% certain.
4. SEARCH FALLBACK: If you are unsure of a direct link, use a YouTube Search URL or Official Docs search URL. 
   - Example YouTube Search: https://www.youtube.com/results?search_query={skill_name.replace(' ', '+')}+tutorial
   - Example Docs Search: https://docs.flutter.dev/search?q={skill_name.replace(' ', '+')}
5. NO BROKEN LINKS: Avoid generating random-looking strings (IDs) that might not exist.

No conversational text outside the JSON block."""

    try:
        logger.info(
            "Skill resources request to Groq | skill=%s | model=llama-3.1-8b-instant",
            skill_name,
        )
        completion = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional educational consultant. "
                        "Always return valid JSON with direct, functional HTTPS URLs."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw_content = completion.choices[0].message.content.strip()
        parsed_data = _parse_json_safely(raw_content)
        return _sanitize_resources(parsed_data)

    except Exception as error:
        _handle_exception(error)


async def analyze_resume_ai(
    resume_text: str,
    job_role: str,
    job_description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a resume against a job role and optional description.
    Returns ATS-friendly structured output with accurate scoring.
    """
    client = get_client()

    # Build context for the AI
    requirements_context = ""
    if job_description and job_description.strip():
        requirements_context = f"\nSpecific Job Requirements/Description:\n{job_description}\n"

    prompt = f"""You are an expert ATS (Applicant Tracking System) resume auditor and senior technical recruiter.

Task: Provide a high-precision analysis of the human resume text against the target role: {job_role}.
{requirements_context}

Resume Text:
{resume_text}

Analyze the match based on skills, experience level, and the specific keywords provided in the job description (if any).

Rules for Match Score Calculation:
- **Be Rigorous**: Do not default to generic scores. A score of 85+ should only be given for near-perfect matches.
- **Scoring Breakdown**:
    - 0-30: Total mismatch or missing 70%+ of core requirements.
    - 31-50: Significant gaps in skills or seniority level.
    - 51-70: Fair match, has core skills but missing several secondary keywords.
    - 71-85: Strong match, missing only minor specific tools or domain experience.
    - 86-100: Exceptional candidate, matches nearly all keywords and experience levels.

Return ONLY valid JSON with this exact structure:
{{
  "matchScore": 0,
  "atsScore": 0,
  "keywordCoverage": 0,
  "detectedSkills": ["skill1", "skill2"],
  "missingSkills": ["missing1", "missing2"],
  "sectionScores": [
    {{
      "section": "Summary",
      "score": 0,
      "details": "Specific ATS-focused observation"
    }}
  ],
  "atsChecks": [
    {{
      "category": "Keyword Alignment",
      "status": "pass|warn|fail",
      "detail": "Specific finding",
      "impact": "Effect on ATS ranking",
      "fix": "Actionable fix"
    }}
  ],
  "strengths": [
    "Specific strength"
  ],
  "improvementPriorities": [
    "Most important ATS fix first"
  ],
  "rewriteSuggestions": [
    "Resume bullet rewrite suggestion"
  ],
  "summary": "2-3 sentence ATS summary explaining exactly why this score was given.",
  "recommendedResources": [
    {{
      "title": "Resource title",
      "url": "https://...",
      "description": "Why this helps",
      "platform": "YouTube / Coursera / Udemy / Official Docs"
    }}
  ]
}}

STRICT JSON RULES:
- matchScore, atsScore, keywordCoverage must be integers 0-100.
- No markdown, no prose, and no conversational text outside the JSON block.
- status in atsChecks must be only pass, warn, or fail.

STRICT RESOURCE URL RULES:
1. Every URL MUST start with https://
2. NO HALLUCINATIONS: Do NOT guess specific sub-page paths or video IDs. 
3. SEARCH FALLBACK: If you are unsure of a direct link, use a YouTube Search URL for that skill.
   - Format: https://www.youtube.com/results?search_query=SKILL+NAME+tutorial
4. PREFER OFFICIAL: Use root documentation URLs if specific paths are unknown (e.g., https://docs.flutter.dev).
"""

    try:
        logger.info(
            "Resume analysis to Groq | role=%s | has_desc=%s | model=llama-3.1-8b-instant",
            job_role,
            bool(job_description),
        )
        completion = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional ATS resume auditor. "
                        "You provide honest, rigorous, and highly specific feedback."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        raw_content = completion.choices[0].message.content.strip()
        parsed_data = _parse_json_safely(raw_content)
        parsed_data["model"] = completion.model
        parsed_data = _sanitize_resources(parsed_data)
        return _normalize_resume_analysis(parsed_data)

    except Exception as error:
        _handle_exception(error)


def _sanitize_resources(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and filter resources to ensure URLs are strictly HTTPS and realistic."""
    blacklist = ["example.com", "placeholder.com", "yourdomain.com", "test.com"]
    
    if "recommendedResources" in data and isinstance(data["recommendedResources"], list):
        valid_resources = []
        for resource in data["recommendedResources"]:
            if not isinstance(resource, dict):
                continue
                
            url = str(resource.get("url", "")).strip()
            
            # Basic validation
            if not url.startswith("https://") or len(url) < 15:
                continue
                
            # Blacklist check
            if any(domain in url.lower() for domain in blacklist):
                continue
            
            # Hallucination pattern check: catch URLs with placeholders like [skill]
            if "[" in url or "{" in url:
                continue
                
            valid_resources.append(resource)
            
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

        logger.error("JSON parsing failed for AI output: %s", raw_text)
        raise GroqServiceError(
            "The AI service returned an unreadable response format.",
            "parse_error",
        )


def _clamp_score(value: Any, default: int = 0) -> int:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(0, min(100, numeric))


def _normalize_status(status: Any) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"pass", "warn", "fail"}:
        return normalized
    return "warn"


def _normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            result.append(text)
    return result


def _normalize_section_scores(section_scores: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    required_sections = [
        "Summary",
        "Experience",
        "Skills",
        "Projects",
        "Education",
        "Formatting",
    ]

    if isinstance(section_scores, list):
        for item in section_scores:
            if not isinstance(item, dict):
                continue
            section_name = str(item.get("section") or "").strip()
            if not section_name:
                continue
            normalized.append(
                {
                    "section": section_name,
                    "score": _clamp_score(item.get("score"), default=50),
                    "details": str(item.get("details") or "").strip() or None,
                }
            )

    existing = {item["section"].lower() for item in normalized}
    for section in required_sections:
        if section.lower() not in existing:
            normalized.append(
                {
                    "section": section,
                    "score": 50,
                    "details": "No detailed feedback available for this section.",
                }
            )

    return normalized


def _normalize_ats_checks(ats_checks: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if isinstance(ats_checks, list):
        for item in ats_checks:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category") or "").strip()
            detail = str(item.get("detail") or "").strip()
            if not category or not detail:
                continue
            normalized.append(
                {
                    "category": category,
                    "status": _normalize_status(item.get("status")),
                    "detail": detail,
                    "impact": str(item.get("impact") or "").strip() or None,
                    "fix": str(item.get("fix") or "").strip() or None,
                }
            )
    return normalized


def _normalize_resume_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    data["matchScore"] = _clamp_score(data.get("matchScore"), default=0)
    data["atsScore"] = _clamp_score(data.get("atsScore"), default=data["matchScore"])
    data["keywordCoverage"] = _clamp_score(
        data.get("keywordCoverage"),
        default=max(0, data["matchScore"] - 5),
    )

    data["detectedSkills"] = _normalize_string_list(data.get("detectedSkills"))
    data["missingSkills"] = _normalize_string_list(data.get("missingSkills"))
    data["sectionScores"] = _normalize_section_scores(data.get("sectionScores"))
    data["atsChecks"] = _normalize_ats_checks(data.get("atsChecks"))
    data["strengths"] = _normalize_string_list(data.get("strengths"))
    data["improvementPriorities"] = _normalize_string_list(
        data.get("improvementPriorities")
    )
    data["rewriteSuggestions"] = _normalize_string_list(data.get("rewriteSuggestions"))

    summary = str(data.get("summary") or "").strip()
    if not summary:
        summary = (
            "Your resume has a fair foundation but needs stronger ATS optimization. "
            "Prioritize keyword alignment, quantified impact, and cleaner section structure."
        )
    data["summary"] = summary
    return data


async def health_check() -> Optional[str]:
    """Diagnostic health check for deployment monitors."""
    try:
        await generate_chat_response("ping")
        return None
    except Exception as error:
        return str(error)
