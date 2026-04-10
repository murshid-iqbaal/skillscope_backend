from pydantic import BaseModel, Field
from typing import List, Optional


class LearningResource(BaseModel):
    title: str = Field(..., description="Title of the resource")
    url: str = Field(..., description="Direct link to the resource")
    description: Optional[str] = Field(default=None, description="Why it's recommended")
    platform: Optional[str] = Field(default=None, description="Platform (e.g. Coursera)")


class ATSSectionScore(BaseModel):
    section: str = Field(..., description="Resume section name")
    score: int = Field(..., ge=0, le=100, description="Section quality score (0-100)")
    details: Optional[str] = Field(default=None, description="Why this section scored this way")


class ATSCheck(BaseModel):
    category: str = Field(..., description="ATS rule/check category")
    status: str = Field(
        ...,
        description="Check status: pass, warn, or fail",
        pattern="^(pass|warn|fail)$",
    )
    detail: str = Field(..., description="Detailed finding")
    impact: Optional[str] = Field(default=None, description="Impact on ATS readability/ranking")
    fix: Optional[str] = Field(default=None, description="Actionable fix")


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
    user_id: Optional[str] = Field(
        default=None,
        description="Optional ID of the user performing the analysis",
    )


class ResumeAnalyzeResponse(BaseModel):
    jobRole: Optional[str] = Field(default=None, description="Echoed job role")
    matchScore: int = Field(..., description="Match score between 0 and 100")
    atsScore: int = Field(default=0, description="ATS-friendliness score between 0 and 100")
    keywordCoverage: int = Field(
        default=0,
        description="Keyword coverage score between 0 and 100",
    )
    detectedSkills: List[str] = Field(..., description="List of skills detected in the resume")
    missingSkills: List[str] = Field(..., description="List of missing skills for the job role")
    sectionScores: List[ATSSectionScore] = Field(
        default_factory=list,
        description="Per-section ATS quality scores",
    )
    atsChecks: List[ATSCheck] = Field(
        default_factory=list,
        description="Detailed ATS checks with status and fixes",
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Strong points found in the resume",
    )
    improvementPriorities: List[str] = Field(
        default_factory=list,
        description="Top prioritized actions to improve ATS performance",
    )
    rewriteSuggestions: List[str] = Field(
        default_factory=list,
        description="Suggested bullet rewrites and phrasing improvements",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Short ATS-style assessment summary",
    )
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
