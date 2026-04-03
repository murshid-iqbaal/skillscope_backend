from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's message or question",
        example="What skills are needed for a Flutter developer?",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID for tracking sessions",
    )


class ChatResponse(BaseModel):
    response: str = Field(
        ...,
        description="The AI-generated response",
    )
    model: str = Field(
        ...,
        description="The Groq model used to generate the response",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Echoed conversation ID if provided",
    )


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error description")
    error_type: str = Field(..., description="Category of error")
