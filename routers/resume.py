import logging
from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form

from models.resume_models import ResumeAnalyzeRequest, ResumeAnalyzeResponse, ErrorResponse
from services.groq_service import (
    analyze_resume_ai,
    GroqServiceError,
    GroqAuthError,
    GroqRateLimitError,
    GroqTimeoutError,
)
from utils.document_parser import parse_resume

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
    },
)
async def analyze_resume(request: ResumeAnalyzeRequest) -> ResumeAnalyzeResponse:
    """
    Primary endpoint for text-based resume analysis using Groq.
    """
    logger.info(f"Text resume analysis request | role={request.job_role}")

    # Text truncation for safety (llama3 limits)
    resume_text = request.resume_text[:12000]

    try:
        # Call standalone async function from groq_service
        result = await analyze_resume_ai(
            resume_text=resume_text,
            job_role=request.job_role,
        )

        return ResumeAnalyzeResponse(
            jobRole=request.job_role,
            matchScore=result.get("matchScore", 0),
            detectedSkills=result.get("detectedSkills", []),
            missingSkills=result.get("missingSkills", []),
            recommendedResources=result.get("recommendedResources", []),
            model=result.get("model"),
        )

    except Exception as e:
        _handle_groq_exception(e)


@router.post(
    "/validate",
    summary="Upload and analyze resume file against a job role",
    description="Accepts PDF/Docx multipart upload and returns analysis in structured JSON.",
)
async def validate_resume(
    job_role: str = Form(..., description="Target job role"),
    resume: UploadFile = File(..., description="Resume file (PDF/Docx)"),
):
    """
    Multipart endpoint for file-based resume analysis.
    Returns format: {"analysis": ResumeAnalyzeResponse} matching Flutter expectations.
    """
    logger.info(f"File resume analysis request received | role={job_role} | file={resume.filename}")
    
    try:
        # 1. Extract text from file
        resume_text = await parse_resume(resume)
        
        # 2. Call AI service
        result = await analyze_resume_ai(
            resume_text=resume_text[:12000],
            job_role=job_role
        )
        
        # 3. Wrap in 'analysis' key (expected by Flutter ResumeApiService)
        # Always log resource URLs for debugging
        resources = result.get("recommendedResources", [])
        for r in resources:
            url = r.get("url", "")
            logger.info(f"Generated resource URL: {url}")

        return {
            "analysis": {
                "jobRole": result.get("jobRole") or job_role,
                "matchScore": result.get("matchScore", 0),
                "detectedSkills": result.get("detectedSkills", []),
                "missingSkills": result.get("missingSkills", []),
                "recommendedResources": resources,
                "model": result.get("model"),
            }
        }
        
    except Exception as e:
        _handle_groq_exception(e)


def _handle_groq_exception(e: Exception):
    """Internal helper to convert service errors to Fast API exceptions."""
    if isinstance(e, (GroqAuthError, GroqRateLimitError, GroqTimeoutError)):
        logger.error(f"Groq API specific error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"detail": f"AI Engine Error: {e.message}", "error_type": e.error_type}
        )
    elif isinstance(e, GroqServiceError):
        logger.error(f"AI service error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"detail": e.message, "error_type": e.error_type}
        )
    else:
        logger.exception(f"Unexpected error in resume analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": "An unexpected error occurred during analysis.",
                "error_type": "internal_error"
            }
        )
