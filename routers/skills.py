import logging
from typing import List
from fastapi import APIRouter
from models.skill_models import Skill

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/skills",
    tags=["Skills Explorer"],
)

# Mock Data for Trending Skills (25+ items)
MOCK_SKILLS = [
    Skill(id="1", name="Flutter", category="Mobile", demand_score=92, growth_rate=24, description="Advanced cross-platform UI toolkit by Google."),
    Skill(id="2", name="Python", category="AI", demand_score=98, growth_rate=15, description="The leading language for AI, data science, and backend development."),
    Skill(id="3", name="React Native", category="Mobile", demand_score=85, growth_rate=12, description="Facebook's cross-platform framework for native-like performance."),
    Skill(id="4", name="Rust", category="Backend", demand_score=88, growth_rate=45, description="Memory-safe language loved by system engineers."),
    Skill(id="5", name="Go (Golang)", category="Backend", demand_score=84, growth_rate=22, description="Efficient, cloud-native backend language by Google."),
    Skill(id="6", name="TensorFlow", category="AI", demand_score=90, growth_rate=18, description="Open-source machine learning platform."),
    Skill(id="7", name="Kubernetes", category="DevOps", demand_score=95, growth_rate=30, description="Container orchestration for massively scalable apps."),
    Skill(id="8", name="Docker", category="DevOps", demand_score=94, growth_rate=10, description="The industry standard for containerization."),
    Skill(id="9", name="TypeScript", category="Web", demand_score=91, growth_rate=28, description="Typed JavaScript that scales."),
    Skill(id="10", name="Next.js", category="Web", demand_score=89, growth_rate=35, description="The React framework for production-grade web apps."),
    Skill(id="11", name="SwiftUI", category="Mobile", demand_score=82, growth_rate=15, description="Apple's modern UI framework for iOS/macOS."),
    Skill(id="12", name="PyTorch", category="AI", demand_score=93, growth_rate=40, description="The preferred deep learning framework for researchers."),
    Skill(id="13", name="PostgreSQL", category="Backend", demand_score=86, growth_rate=8, description="The world's most advanced open-source database."),
    Skill(id="14", name="Nginx", category="DevOps", demand_score=80, growth_rate=5, description="High-performance web server and reverse proxy."),
    Skill(id="15", name="Firebase", category="Mobile", demand_score=83, growth_rate=12, description="Google's backend-as-a-service platform."),
    Skill(id="16", name="AWS Lambda", category="DevOps", demand_score=87, growth_rate=20, description="Serverless computing by Amazon."),
    Skill(id="17", name="Django", category="Backend", demand_score=81, growth_rate=7, description="The web framework for perfectionists with deadlines."),
    Skill(id="18", name="GraphQL", category="Web", demand_score=78, growth_rate=15, description="A query language for APIs."),
    Skill(id="19", name="Tailwind CSS", category="Web", demand_score=92, growth_rate=50, description="Utility-first CSS framework for rapid styling."),
    Skill(id="20", name="Zustand", category="Web", demand_score=75, growth_rate=40, description="Small, fast, and scalable bearbones state-management."),
    Skill(id="21", name="LangChain", category="AI", demand_score=97, growth_rate=200, description="Framework for developing LLM-powered applications."),
    Skill(id="22", name="Redis", category="Backend", demand_score=85, growth_rate=14, description="In-memory data structure store, used as cache."),
    Skill(id="23", name="Kotlin", category="Mobile", demand_score=81, growth_rate=9, description="The modern language for Android development."),
    Skill(id="24", name="Terraform", category="DevOps", demand_score=89, growth_rate=25, description="Infrastructure as Code (IaC) tool."),
    Skill(id="25", name="FastAPI", category="Backend", demand_score=90, growth_rate=60, description="Modern, fast (high-performance), web framework for building APIs."),
]

@router.get(
    "/trending",
    response_model=List[Skill],
    summary="Fetch real-time trending skills",
    description="Returns a curated list of skills sorted by demand score (descending)."
)
async def get_trending_skills():
    """
    Returns the trending skills list. 
    Sorted by demand_score DESC.
    """
    logger.info("Fetching trending skills...")
    # Sort by demand score (DESC)
    sorted_skills = sorted(MOCK_SKILLS, key=lambda x: x.demand_score, reverse=True)
    return sorted_skills
