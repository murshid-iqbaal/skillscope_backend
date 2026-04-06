import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from routers.chat import router as chat_router
from routers.resume import router as resume_router
from routers.skills import router as skills_router
from routers.resources import router as resources_router



# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Lifespan: startup / shutdown hooks
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("SkillScope AI backend starting up...")
    try:
        settings.validate()
        logger.info("✅ Config validated — GROQ_API_KEY is present.")
    except ValueError as e:
        logger.critical(f"❌ Configuration error: {e}")
        sys.exit(1)

    yield

    # Shutdown
    logger.info("SkillScope AI backend shutting down.")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=(
        "Backend API for SkillScope AI — a career and tech learning assistant. "
        "Powered by Groq for ultra-fast AI responses."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# CORS — allow all origins for mobile app access
# ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Lock this down in production if needed
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Routers
# ──────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(resume_router)
app.include_router(skills_router)
app.include_router(resources_router)




# ──────────────────────────────────────────────
# Built-in health + info routes
# ──────────────────────────────────────────────
@app.get("/", tags=["System"], summary="Root — API info")
async def root():
    return {
        "message": "Welcome to SkillScope AI API",
        "status": "online",
        "docs": "/docs",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health",
            "health_groq": "GET /health/groq"
        }
    }



@app.get("/health", tags=["System"], summary="Health check")
async def health():
    """
    Quick liveness probe used by Render and load balancers.
    Returns 200 if the server is running.
    """
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "service": settings.APP_TITLE},
    )


from models.resume_models import JobRoleSkills, LearningResourceRequest
from typing import List

@app.get("/job-roles", response_model=List[JobRoleSkills], tags=["Data Models"])
async def get_job_roles():
    """Mock endpoint returning common tech job roles and required skills."""
    return [
        {"role": "Flutter Developer", "skills": ["Dart", "Flutter", "State Management", "Firebase"]},
        {"role": "Backend Developer", "skills": ["Python", "FastAPI", "SQL", "Docker"]},
        {"role": "Data Scientist", "skills": ["Python", "Machine Learning", "Pandas", "Scikit-Learn"]},
        {"role": "Frontend Developer", "skills": ["JavaScript", "React", "CSS", "HTML"]},
        {"role": "DevOps Engineer", "skills": ["AWS", "Kubernetes", "Docker", "CI/CD"]},
    ]


@app.post("/learning-resources", tags=["Data Models"])
async def get_learning_resources(request: LearningResourceRequest):
    """Mock endpoint returning learning resources for requested skills."""
    # Build a simple mock response based on the requested skills
    resources = []
    for skill in request.skills:
        resources.append({
            "title": f"Mastering {skill}",
            "url": f"https://example.com/learn-{skill.lower().replace(' ', '-')}",
            "description": f"A comprehensive guide to {skill}",
            "platform": "SkillScope Academy"
        })
    return resources


@app.get("/health/groq", tags=["System"], summary="Groq API connectivity check")
async def health_groq():
    """
    Deep health check — tests live connectivity to Groq API.
    Use this to verify the API key and Groq availability during deployment.
    """
    from services.groq_service import health_check
    error = await health_check()
    if error:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "groq_error": error},
        )
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "groq": "reachable"},
    )
