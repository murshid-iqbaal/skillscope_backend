from pydantic import BaseModel, Field
from typing import List, Optional


class LearningResource(BaseModel):
    title: str = Field(..., description="Title of the resource")
    url: str = Field(..., description="Direct link to the resource")
    description: Optional[str] = Field(default=None, description="Why it's recommended")
    platform: Optional[str] = Field(default=None, description="Platform (e.g. Coursera)")


class ResumeAnalyzeRequest(BaseModel):
    job_role: str = Field(
        ...,
        description="The target job role to analyze the resume against",
        example="Flutter Developer",
    )
    resume_text: str = Field(
        ...,
        description="The raw text extracted from the resume",
        max_length=15000,
    )


class ResumeAnalyzeResponse(BaseModel):
    jobRole: Optional[str] = Field(default=None, description="Echoed job role")
    matchScore: int = Field(..., description="Match score between 0 and 100")
    detectedSkills: List[str] = Field(..., description="List of skills detected in the resume")
    missingSkills: List[str] = Field(..., description="List of missing skills for the job role")
    recommendedResources: List[LearningResource] = Field(
        default_factory=list, 
        description="List of AI recommended learning resources"
    )
    model: Optional[str] = Field(default=None, description="The model used for analysis")


class JobRoleSkills(BaseModel):
    role: str = Field(..., description="The job role name")
    skills: List[str] = Field(default_factory=list, description="List of required skills")


class LearningResourceRequest(BaseModel):
    skills: List[str] = Field(..., description="Skills to fetch resources for")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error description")
    error_type: str = Field(..., description="Category of error")
