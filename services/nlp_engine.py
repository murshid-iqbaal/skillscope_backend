import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Basic dictionary of skills for fallback logic
# In a real-world scenario, this could be fetched from a database.
_SKILL_DATABASE = {
    "flutter developer": [
        "flutter", "dart", "widget", "state management", "riverpod", "bloc",
        "rest api", "json", "firebase", "sqlite", "git", "ci/cd", "unit testing",
        "material design", "cupertino", "ios", "android", "play store", "app store"
    ],
    "backend developer": [
        "python", "fastapi", "flask", "django", "nodejs", "express", "postgresql",
        "mysql", "mongodb", "redis", "docker", "kubernetes", "aws", "azure", "gcp",
        "rest", "graphql", "microservices", "unit testing", "git"
    ],
    "frontend developer": [
        "javascript", "typescript", "react", "vue", "angular", "html", "css", "sass",
        "tailwind", "bootstrap", "webpack", "vite", "unit testing", "git", "redux",
        "responsive design", "ui/ux", "nextjs"
    ],
    "data scientist": [
        "python", "r", "sql", "pandas", "numpy", "scikit-learn", "tensorflow",
        "pytorch", "machine learning", "deep learning", "nlp", "statistics",
        "jupyter", "matplotlib", "seaborn", "tableau", "big data"
    ]
}

class NLPEngine:
    """
    Local NLP fallback engine to analyze resumes based on keyword matching.
    Used when the AI service is unavailable.
    """

    def analyze_resume(self, resume_text: str, job_role: str) -> Dict[str, Any]:
        """
        Perform a simple keyword-based analysis of the resume.
        """
        logger.info(f"Using local NLP fallback for resume analysis | role={job_role}")
        
        # Clean and normalize text
        resume_text_lower = resume_text.lower()
        role_lower = job_role.lower()
        
        # Get target skills for the role
        target_skills = _SKILL_DATABASE.get(role_lower, [])
        if not target_skills:
            # If role is unknown, try to extract common tech skills from text
            # but we'll default to a basic generic set for demo purposes.
            target_skills = ["git", "docker", "sql", "api", "testing", "agile"]
        
        detected_skills = []
        missing_skills = []
        
        for skill in target_skills:
            # Use regex to find whole words for better accuracy
            pattern = re.compile(rf"\b{re.escape(skill)}\b", re.IGNORECASE)
            if pattern.search(resume_text_lower):
                detected_skills.append(skill)
            else:
                missing_skills.append(skill)
        
        # Calculate match score based on detected skills vs total target skills
        if target_skills:
            match_score = int((len(detected_skills) / len(target_skills)) * 100)
        else:
            match_score = 0
            
        return {
            "matchScore": match_score,
            "detectedSkills": detected_skills,
            "missingSkills": missing_skills,
            "suggestions": "The AI service is currently unavailable. This result is based on local keyword matching. Consider adding more relevant keywords found in the job description to your resume.",
            "model": "local-nlp-engine-v1"
        }

# Module-level singleton
nlp_engine = NLPEngine()
