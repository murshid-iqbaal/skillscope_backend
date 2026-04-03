import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from routers.chat import router as chat_router

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


# ──────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────
app.include_router(chat_router)


# ──────────────────────────────────────────────
# Built-in health + info routes
# ──────────────────────────────────────────────
@app.get("/", tags=["System"], summary="Root — API info")
async def root():
    return {
        "name": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "chat_endpoint": "POST /chat",
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
