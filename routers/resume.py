import logging
from fastapi import APIRouter, HTTPException, status

from models.resume_models import ResumeAnalyzeRequest, ResumeAnalyzeResponse, ErrorResponse
from services.groq_service import (
    analyze_resume_ai,
    GroqServiceError,
    GroqAuthError,
    GroqRateLimitError,
    GroqTimeoutError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/resume",
    tags=["Resume Validator"],
)


@router.post(
    "/analyze",
    response_model=ResumeAnalyzeResponse,
    summary="Analyze resume text against a job role",
    description=(
        "Uses Groq AI to analyze matching capability for a specific job role. "
        "Detects skills, missing keywords, and provides improvement suggestions."
    ),
    responses={
        200: {"description": "Resume analyzed successfully"},
        401: {"model": ErrorResponse, "description": "Auth error"},
        429: {"model": ErrorResponse, "description": "Rate limit"},
        500: {"model": ErrorResponse, "description": "AI failure"},
    },
)
async def analyze_resume(request: ResumeAnalyzeRequest) -> ResumeAnalyzeResponse:
    """
    Primary endpoint for AI-based resume analysis using Groq.
    """
    logger.info(f"Resume analysis request | role={request.job_role}")

    # Text truncation for safety (llama3 limits)
    resume_text = request.resume_text[:12000]

    try:
        # Call standalone async function from groq_service
        result = await analyze_resume_ai(
            resume_text=resume_text,
            job_role=request.job_role,
        )

        return ResumeAnalyzeResponse(
            matchScore=result.get("matchScore", 0),
            detectedSkills=result.get("detectedSkills", []),
            missingSkills=result.get("missingSkills", []),
            suggestions=result.get("suggestions", "No suggestions available."),
            model=result.get("model"),
        )

    except (GroqAuthError, GroqRateLimitError, GroqTimeoutError) as e:
        logger.error(f"Groq-specific error in resume analysis: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"detail": f"AI Service Error: {e.message}", "error_type": e.error_type}
        )

    except GroqServiceError as e:
        logger.error(f"General AI Service Error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"detail": e.message, "error_type": e.error_type}
        )

    except Exception as e:
        logger.exception(f"Unexpected error in resume analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": "An unexpected error occurred during analysis.",
                "error_type": "internal_error"
            }
        )
