import logging
from fastapi import APIRouter, HTTPException, status

from models.chat_models import ChatRequest, ChatResponse, ErrorResponse
from services.groq_service import (
    generate_response,
    GroqAuthError,
    GroqRateLimitError,
    GroqTimeoutError,
    GroqNetworkError,
    GroqServiceError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a message to SkillScope AI",
    description=(
        "Send a career or tech question and receive an AI-generated response "
        "powered by Groq. Supports questions about skills, job roles, learning paths, and technologies."
    ),
    responses={
        200: {"description": "AI response returned successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        408: {"model": ErrorResponse, "description": "Request timeout"},
        503: {"model": ErrorResponse, "description": "AI service unavailable"},
        500: {"model": ErrorResponse, "description": "Unexpected server error"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint. Accepts a user message and returns an AI response.
    """
    logger.info(f"Chat request received | msg_len={len(request.message)}")

    try:
        result = await generate_response(message=request.message)

        return ChatResponse(
            response=result["response"],
            model=result["model"],
            conversation_id=request.conversation_id,
        )

    except GroqAuthError as e:
        logger.error(f"Auth error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"detail": e.message, "error_type": e.error_type},
        )

    except GroqRateLimitError as e:
        logger.warning(f"Rate limit hit: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"detail": e.message, "error_type": e.error_type},
        )

    except GroqTimeoutError as e:
        logger.warning(f"Timeout: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail={"detail": e.message, "error_type": e.error_type},
        )

    except GroqNetworkError as e:
        logger.error(f"Network error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"detail": e.message, "error_type": e.error_type},
        )

    except GroqServiceError as e:
        logger.error(f"Groq service error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"detail": e.message, "error_type": e.error_type},
        )

    except Exception as e:
        logger.exception(f"Unhandled exception in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": "An unexpected error occurred. Please try again later.",
                "error_type": "internal_error",
            },
        )
