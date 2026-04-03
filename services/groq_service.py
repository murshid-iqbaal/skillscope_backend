import httpx
import logging
from typing import Optional

from core.config import settings
from utils.prompt_builder import build_prompt

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


class GroqNetworkError(GroqServiceError):
    def __init__(self, detail: str = ""):
        super().__init__(
            message=f"Network error while connecting to Groq API. {detail}".strip(),
            error_type="network_error",
        )


async def generate_response(
    message: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> dict:
    """
    Call Groq API with the user message and return the AI response text.

    Returns:
        dict with keys: 'response' (str) and 'model' (str)

    Raises:
        GroqServiceError and its subclasses on failure.
    """
    if not settings.GROQ_API_KEY:
        raise GroqAuthError()

    messages = build_prompt(message)

    payload = {
        "model": settings.GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
            logger.info(f"Sending request to Groq | model={settings.GROQ_MODEL} | msg_len={len(message)}")
            resp = await client.post(
                settings.GROQ_API_URL,
                json=payload,
                headers=headers,
            )

        _handle_groq_status(resp)

        data = resp.json()
        response_text = _extract_response_text(data)
        model_used = data.get("model", settings.GROQ_MODEL)

        logger.info(f"Groq response received | model={model_used} | resp_len={len(response_text)}")
        return {"response": response_text, "model": model_used}

    except httpx.TimeoutException:
        logger.error("Groq API request timed out.")
        raise GroqTimeoutError()

    except httpx.ConnectError as e:
        logger.error(f"Groq API connection error: {e}")
        raise GroqNetworkError(str(e))

    except httpx.RequestError as e:
        logger.error(f"Groq API request error: {e}")
        raise GroqNetworkError(str(e))

    except GroqServiceError:
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in Groq service: {e}")
        raise GroqServiceError(
            message="An unexpected error occurred while processing your request.",
            error_type="unexpected_error",
        )


def _handle_groq_status(response: httpx.Response) -> None:
    """Map Groq HTTP status codes to typed exceptions."""
    if response.status_code == 200:
        return

    if response.status_code == 401:
        raise GroqAuthError()

    if response.status_code == 429:
        raise GroqRateLimitError()

    if response.status_code == 408:
        raise GroqTimeoutError()

    # Try to extract Groq error message from body
    try:
        error_body = response.json()
        detail = error_body.get("error", {}).get("message", "Unknown error")
    except Exception:
        detail = f"HTTP {response.status_code}"

    logger.error(f"Groq API error {response.status_code}: {detail}")
    raise GroqServiceError(
        message=f"Groq API returned an error: {detail}",
        error_type="api_error",
    )


def _extract_response_text(data: dict) -> str:
    """Safely extract the assistant message from Groq's response."""
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Failed to parse Groq response structure: {data}")
        raise GroqServiceError(
            message="Received an unexpected response format from Groq API.",
            error_type="parse_error",
        )


async def health_check() -> Optional[str]:
    """
    Lightweight check to verify Groq API is reachable.
    Returns None on success, error string on failure.
    """
    try:
        result = await generate_response("Say 'OK' in one word.", max_tokens=10)
        return None
    except GroqServiceError as e:
        return e.message
    except Exception as e:
        return str(e)
