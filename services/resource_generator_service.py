import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from services.groq_service import get_client, _parse_json_safely
from models.resume_models import LearningResource

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# In-memory Caching
# ──────────────────────────────────────────────
_resource_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = timedelta(hours=24)

def _get_cached_resources(skill_name: str) -> Optional[List[Dict[str, Any]]]:
    """Retrieve resources from cache if valid and not expired."""
    key = skill_name.lower().strip()
    if key in _resource_cache:
        cached_data = _resource_cache[key]
        if datetime.now() < cached_data["expiry"]:
            logger.info(f"Cache hit for skill: {skill_name}")
            return cached_data["resources"]
    return None

def _set_cached_resources(skill_name: str, resources: List[Dict[str, Any]]):
    """Store resources in cache with expiry."""
    key = skill_name.lower().strip()
    _resource_cache[key] = {
        "resources": resources,
        "expiry": datetime.now() + CACHE_TTL
    }

# ──────────────────────────────────────────────
# Resource Generator Logic
# ──────────────────────────────────────────────

async def generate_resources_for_skill(skill_name: str) -> List[Dict[str, Any]]:
    """
    Generate high-quality learning resources for a skill using Groq API.
    Includes validation and fallback logic.
    """
    # 1. Check Cache
    cached = _get_cached_resources(skill_name)
    if cached:
        return cached

    # 2. Call Groq API
    client = get_client()
    prompt = f"""
You are an expert developer mentor.

Generate the best learning resources for the following skill.

Skill: {skill_name}

Return ONLY valid JSON:

{{
  "resources": [
    {{
      "title": "string",
      "url": "FULL VALID HTTPS URL",
      "description": "short description",
      "platform": "YouTube / Documentation / Course"
    }}
  ]
}}

STRICT RULES:

• URL must start with https://
• URLs must be real, working, and direct links
• For YouTube → use full video link: https://www.youtube.com/watch?v=VIDEO_ID
• For documentation → use official docs
• For courses → use real platforms

DO NOT:

• return search queries
• return incomplete URLs
• return plain text
• return invalid links

Return only JSON, no explanation.
"""

    try:
        logger.info(f"Generating resources for {skill_name} using llama3-70b-8192")
        completion = await client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a professional technology mentor. Always return valid JSON with curated learning resources."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        raw_content = completion.choices[0].message.content.strip()
        parsed_data = _parse_json_safely(raw_content)
        resources = parsed_data.get("resources", [])

        # 3. Validate and fallback
        validated_resources = []
        for res in resources:
            title = res.get("title", f"Learn {skill_name}")
            url = res.get("url", "").strip()
            desc = res.get("description", "")
            platform = res.get("platform", "Web")

            # Validation logic
            if not url or not url.startswith("https://") or len(url) < 15:
                logger.warning(f"Invalid URL detected for {skill_name}: {url}. Using fallback.")
                url = f"https://www.google.com/search?q={skill_name}+learning+resources"
                platform = "Search Engine"

            validated_resources.append({
                "title": title,
                "url": url,
                "description": desc,
                "platform": platform
            })

        # 4. Handle empty response from AI
        if not validated_resources:
            logger.info(f"No resources returned for {skill_name}, providing base fallback.")
            validated_resources = [{
                "title": f"Mastering {skill_name}",
                "url": f"https://www.google.com/search?q={skill_name}+official+documentation",
                "description": f"Explore curated resources to master {skill_name}.",
                "platform": "Google Search"
            }]

        # 5. Cache and Return
        _set_cached_resources(skill_name, validated_resources)
        return validated_resources

    except Exception as e:
        logger.error(f"Error generating resources for {skill_name}: {e}")
        # Return graceful fallback on API failure
        return [{
            "title": f"Explore {skill_name}",
            "url": f"https://www.google.com/search?q={skill_name}+tutorial",
            "description": "Discover high-quality tutorials and guides for this skill.",
            "platform": "Search"
        }]
