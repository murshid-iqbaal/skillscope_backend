import json
import logging
import re
from typing import Optional

from groq import AsyncGroq
from core.config import settings

logger = logging.getLogger(__name__)


class GroqServiceError(Exception):
    """Base exception for Groq service errors."""
    def __init__(self, message: str, error_type: str = "groq_error"):
        self.message = message
        self.error_type = error_type
        super().__init__(message)


class GroqAuthError(GroqServiceError):
    def __init__(self):
        super().__init__(
            message="Invalid or missing Groq API key. Check your GROQ_API_KEY environment variable.",
            error_type="auth_error",
        )


class GroqRateLimitError(GroqServiceError):
    def __init__(self):
        super().__init__(
            message="Groq API rate limit reached. Please try again in a moment.",
            error_type="rate_limit_error",
        )


class GroqTimeoutError(GroqServiceError):
    def __init__(self):
        super().__init__(
            message="Request to Groq API timed out. Please try again.",
            error_type="timeout_error",
        )


class GroqService:
    """
    Service for interacting with the Groq API for various AI tasks.
    """

    def __init__(self) -> None:
        self._client: Optional[AsyncGroq] = None

    @property
    def client(self) -> AsyncGroq:
        if self._client is None:
            if not settings.GROQ_API_KEY:
                raise GroqAuthError()
            self._client = AsyncGroq(
                api_key=settings.GROQ_API_KEY,
                timeout=settings.REQUEST_TIMEOUT,
            )
        return self._client

    async def generate_response(
        self,
        message: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Call Groq API with the user message and return the AI response text.
        """
        try:
            logger.info(f"Sending chat request to Groq | model={settings.GROQ_MODEL}")
            completion = await self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": message}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            response_text = completion.choices[0].message.content.strip()
            model_used = completion.model

            logger.info(f"Groq response received | model={model_used}")
            return {"response": response_text, "model": model_used}


        except Exception as e:
            self._handle_exception(e)

    async def analyze_resume_ai(
        self,
        resume_text: str,
        job_role: str,
    ) -> dict:
        """
        Analyze a resume against a job role using Groq AI.
        """
        prompt = f"""You are an expert ATS resume analyzer.

Analyze the following resume for the given job role.

Job Role: {job_role}

Resume:
{resume_text}

Return ONLY valid JSON with the following structure:

{{
"matchScore": number (0-100),
"detectedSkills": [list of skills],
"missingSkills": [list of missing skills],
"suggestions": "short improvement suggestions"
}}

Do not include any explanation outside JSON."""

        try:
            logger.info(f"Sending resume analysis request to Groq | role={job_role} | model=llama3-70b-8192")
            completion = await self.client.chat.completions.create(
                model="llama3-70b-8192",

                messages=[{"role": "system", "content": "You are a professional ATS analyzer."},
                          {"role": "user", "content": prompt}],
                temperature=0.2, # Lower temperature for strictly formatted output
                response_format={"type": "json_object"},
            )
            
            raw_content = completion.choices[0].message.content.strip()
            logger.info("Groq resume analysis received.")
            
            parsed_data = self._parse_json_safely(raw_content)
            parsed_data["model"] = completion.model
            return parsed_data

        except Exception as e:
            self._handle_exception(e)

    def _parse_json_safely(self, raw_text: str) -> dict:
        """Extract and parse JSON from model output."""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback: extract using regex if the model included extra text
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            
            logger.error(f"Failed to parse JSON from AI response: {raw_text}")
            raise GroqServiceError(
                message="AI returned an invalid response format.",
                error_type="parse_error",
            )

    def _handle_exception(self, e: Exception) -> None:
        """Map Groq API exceptions to internal service errors."""
        error_str = str(e)
        logger.error(f"Groq API error: {error_str}")
        
        if "rate_limit" in error_str.lower():
            raise GroqRateLimitError()
        elif "timeout" in error_str.lower():
            raise GroqTimeoutError()
        elif "authentication" in error_str.lower() or "api_key" in error_str.lower():
            raise GroqAuthError()
        else:
            raise GroqServiceError(
                message=f"An error occurred while contacting the AI service: {error_str}",
                error_type="api_error",
            )

    async def health_check(self) -> Optional[str]:
        """Verify internal connectivity to Groq."""
        try:
            await self.generate_response("Say OK", max_tokens=10)
            return None
        except Exception as e:
            return str(e)


# Module-level singleton
groq_service = GroqService()
