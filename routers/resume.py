import logging
from fastapi import APIRouter, HTTPException, status

from models.resume_models import ResumeAnalyzeRequest, ResumeAnalyzeResponse, ErrorResponse
from services.groq_service import groq_service, GroqServiceError
from services.nlp_engine import nlp_engine

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
        "Uses Groq AI to analyze a resume's match for a specific job role. "
        "Detects skills, identifies missing ones, and provides improvement suggestions."
    ),
    responses={
        200: {"description": "Resume analyzed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Unauthorized (API Key error)"},
        429: {"model": ErrorResponse, "description": "Rate limit reached"},
        500: {"model": ErrorResponse, "description": "AI analysis error"},
    },
)
async def analyze_resume(request: ResumeAnalyzeRequest) -> ResumeAnalyzeResponse:
    """
    Primary endpoint for AI-based resume analysis.
    """
    logger.info(f"Resume analysis request | role={request.job_role} | text_len={len(request.resume_text)}")
    
    # 1. Truncate text to avoid model limits
    # Most resumes won't exceed this, but we'll cap it at 12,000 chars for safety.
    resume_text = request.resume_text[:12000]
    
    try:
        # 2. Attempt AI-based analysis
        result = await groq_service.analyze_resume_ai(
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

    except GroqServiceError as e:
        logger.warning(f"AI Analysis failed, falling back to local NLP: {e.message}")
        
        # 3. Fallback to local NLP engine if Groq fails
        try:
            fallback_result = nlp_engine.analyze_resume(
                resume_text=resume_text,
                job_role=request.job_role,
            )
            
            return ResumeAnalyzeResponse(
                matchScore=fallback_result["matchScore"],
                detectedSkills=fallback_result["detectedSkills"],
                missingSkills=fallback_result["missingSkills"],
                suggestions=fallback_result["suggestions"],
                model=fallback_result["model"],
            )
            
        except Exception as fallback_err:
            logger.error(f"Fallback NLP analysis failed: {fallback_err}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "detail": "Both AI and fallback analysis engines failed.",
                    "error_type": "all_engines_failed"
                }
            )

    except Exception as e:
        logger.exception(f"Unexpected error in resume analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": "An unexpected error occurred during resume analysis.",
                "error_type": "internal_error"
            }
        )
