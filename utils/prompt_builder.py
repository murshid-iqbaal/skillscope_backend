from typing import List, Dict


SYSTEM_PROMPT = """You are SkillScope AI, an expert career and technology assistant embedded in a mobile app.

Your purpose is to help users:
• Learn technical skills (Flutter, Python, React, DevOps, AI/ML, and more)
• Choose and plan career paths in the tech industry
• Understand job roles, responsibilities, and required qualifications
• Explore technologies, tools, frameworks, and industry trends
• Prepare for technical interviews and assessments
• Get actionable, step-by-step learning roadmaps

Guidelines:
- Always give clear, concise, and actionable answers
- Use bullet points or numbered lists for structured information
- When listing skills or steps, be specific (e.g., mention package names, tools, or platforms)
- Tailor answers to different experience levels when context is provided
- If a question is outside career/tech scope, politely redirect to your area of expertise
- Avoid vague answers — always provide concrete next steps or resources when relevant"""


def build_prompt(user_message: str) -> List[Dict[str, str]]:
    """
    Build the messages array for the Groq chat completion API.

    Returns a list of message dicts with system and user roles.
    """
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message.strip(),
        },
    ]


def build_conversation_prompt(
    user_message: str,
    history: List[Dict[str, str]] | None = None,
) -> List[Dict[str, str]]:
    """
    Build prompt with optional conversation history for multi-turn support.
    History should be a list of {"role": "user"/"assistant", "content": "..."} dicts.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_message.strip()})
    return messages
