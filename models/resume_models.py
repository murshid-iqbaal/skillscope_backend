from pydantic import BaseModel, Field
from typing import List, Optional


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
    matchScore: int = Field(..., description="Match score between 0 and 100")
    detectedSkills: List[str] = Field(..., description="List of skills detected in the resume")
    missingSkills: List[str] = Field(..., description="List of missing skills for the job role")
    suggestions: str = Field(..., description="Short improvement suggestions")
    model: Optional[str] = Field(default=None, description="The model used for analysis")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error description")
    error_type: str = Field(..., description="Category of error")
