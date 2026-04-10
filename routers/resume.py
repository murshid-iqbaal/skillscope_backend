import logging
from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form, BackgroundTasks

from models.resume_models import ResumeAnalyzeRequest, ResumeAnalyzeResponse, ErrorResponse
from services.groq_service import (
    analyze_resume_ai,
    GroqServiceError,
    GroqAuthError,
    GroqRateLimitError,
    GroqTimeoutError,
)
from utils.document_parser import parse_resume
from core.supabase import get_supabase
from typing import Optional

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
async def analyze_resume(
    request: ResumeAnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> ResumeAnalyzeResponse:
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

        # 2. Save to database for analytics (Background Task)
        background_tasks.add_task(
            _save_analysis_to_db,
            user_id=request.user_id,
            match_score=result.get("matchScore", 0),
            detected_skills=result.get("detected_skills", result.get("detectedSkills", [])),
            missing_skills=result.get("missing_skills", result.get("missingSkills", [])),
        )

        return ResumeAnalyzeResponse(
            jobRole=request.job_role,
            matchScore=result.get("matchScore", 0),
            atsScore=result.get("atsScore", 0),
            keywordCoverage=result.get("keywordCoverage", 0),
            detectedSkills=result.get("detectedSkills", []),
            missingSkills=result.get("missingSkills", []),
            sectionScores=result.get("sectionScores", []),
            atsChecks=result.get("atsChecks", []),
            strengths=result.get("strengths", []),
            improvementPriorities=result.get("improvementPriorities", []),
            rewriteSuggestions=result.get("rewriteSuggestions", []),
            summary=result.get("summary"),
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
    background_tasks: BackgroundTasks = None,
    user_id: Optional[str] = Form(None, description="Optional user ID"),
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

        # 3. Save to database for analytics (Background Task)
        if background_tasks:
            background_tasks.add_task(
                _save_analysis_to_db,
                user_id=user_id,
                match_score=result.get("matchScore", 0),
                detected_skills=result.get("detected_skills", result.get("detectedSkills", [])),
                missing_skills=result.get("missing_skills", result.get("missingSkills", [])),
            )

        return {
            "analysis": {
                "jobRole": result.get("jobRole") or job_role,
                "matchScore": result.get("matchScore", 0),
                "atsScore": result.get("atsScore", 0),
                "keywordCoverage": result.get("keywordCoverage", 0),
                "detectedSkills": result.get("detectedSkills", []),
                "missingSkills": result.get("missingSkills", []),
                "sectionScores": result.get("sectionScores", []),
                "atsChecks": result.get("atsChecks", []),
                "strengths": result.get("strengths", []),
                "improvementPriorities": result.get("improvementPriorities", []),
                "rewriteSuggestions": result.get("rewriteSuggestions", []),
                "summary": result.get("summary"),
                "recommendedResources": resources,
                "model": result.get("model"),
            }
        }
        
    except Exception as e:
        _handle_groq_exception(e)


def _save_analysis_to_db(
    user_id: Optional[str],
    match_score: int,
    detected_skills: list,
    missing_skills: list,
):
    """
    Persist analysis results to Supabase for admin dashboard tracking.
    Fails gracefully to not block the user response.
    """
    supabase = get_supabase()
    if not supabase:
        logger.warning("Supabase client not available, skipping analysis persistence.")
        return

    try:
        data = {
            "match_score": match_score,
            "detected_skills": detected_skills,
            "missing_skills": missing_skills,
        }
        if user_id:
            data["user_id"] = user_id

        # Insert into the resume_analysis table
        supabase.table("resume_analysis").insert(data).execute()
        logger.info(f"Successfully persisted resume analysis for user={user_id or 'anonymous'}")
    except Exception as e:
        logger.error(f"Failed to persist resume analysis to DB: {e}")


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
