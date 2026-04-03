from pydantic import BaseModel, Field
from typing import Optional


class Skill(BaseModel):
    id: str = Field(..., description="Unique skill ID")
    name: str = Field(..., description="Skill name (e.g. Flutter)")
    category: str = Field(..., description="Skill category (e.g. Mobile)")
    demand_score: int = Field(..., description="Current demand score (0-100)")
    growth_rate: int = Field(..., description="Year-over-year growth percentage")
    description: Optional[str] = Field(default=None, description="Short skill description")
