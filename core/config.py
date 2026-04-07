import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    APP_TITLE: str = "SkillScope AI Backend"
    APP_VERSION: str = "1.0.0"

    # Supabase (NEW)
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

    def validate(self):
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Check your .env file.")
        if not self.SUPABASE_URL or not self.SUPABASE_KEY:
            # We'll allow it for now but log a warning if needed
            print("Warning: SUPABASE_URL or SUPABASE_KEY is missing. Database resources will not be available.")

settings = Settings()
