import logging
from typing import List, Dict, Any
from core.supabase import get_supabase
from services.groq_service import generate_skill_resources

logger = logging.getLogger(__name__)

async def get_learning_resources(skill_name: str) -> Dict[str, Any]:
    """
    Fetch learning resources for a skill.
    Tries Supabase first, then falls back to Groq AI.
    """
    resources = await get_db_learning_resources(skill_name)
    
    if not resources:
        logger.info(f"No resources found in DB for {skill_name}. Falling back to Groq AI.")
        try:
            return await generate_skill_resources(skill_name)
        except Exception as e:
            logger.error(f"Failed to generate resources for {skill_name} via AI: {e}")
            return {"skill": skill_name, "recommendedResources": []}
            
    return {
        "skill": skill_name,
        "recommendedResources": resources,
        "source": "database"
    }

async def get_db_learning_resources(skill_name: str) -> List[Dict[str, Any]]:
    """
    Query the learning_resources table in Supabase.
    """
    supabase = get_supabase()
    if not supabase:
        return []
        
    try:
        # Case-insensitive match for skill_name
        response = supabase.table("learning_resources") \
            .select("title, url, type, platform") \
            .ilike("skill_name", skill_name) \
            .execute()
            
        if response.data:
            logger.info(f"Found {len(response.data)} resources for {skill_name} in DB.")
            return response.data
    except Exception as e:
        logger.error(f"Error fetching resources from DB for {skill_name}: {e}")
        
    return []
