import logging
from typing import List
from fastapi import APIRouter, HTTPException

from services.resource_generator_service import generate_resources_for_skill
from models.resume_models import LearningResource

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/resources",
    tags=["Resource Generator"],
)

@router.get(
    "/{skill_name}",
    response_model=List[LearningResource],
    summary="Generate learning resources for a skill",
    description="Uses AI to curate high-quality documentation, videos, and courses for a specific skill."
)
async def get_resources_for_skill(skill_name: str):
    """
    Fetch or generate learning resources for the requested skill.
    Returns a list of validate LearningResource objects.
    """
    logger.info(f"API Request: Fetch resources for {skill_name}")
    try:
        resources = await generate_resources_for_skill(skill_name)
        if not resources:
            raise HTTPException(
                status_code=404, 
                detail=f"No resources found for skill: {skill_name}"
            )
        return resources
    except Exception as e:
        logger.error(f"Router error fetching resources for {skill_name}: {e}")
        # The service already has fallbacks, so this is for catastrophic failure
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating learning resources."
        )
