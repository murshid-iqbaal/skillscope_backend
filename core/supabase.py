import logging
from supabase import create_client, Client
from core.config import settings

logger = logging.getLogger(__name__)

_client: Client = None

def get_supabase() -> Client:
    """
    Initialize and return the Supabase client.
    Returns None if credentials are missing.
    """
    global _client
    if _client is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            logger.warning("Supabase credentials missing. Supabase features will be disabled.")
            return None
            
        try:
            _client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            logger.info("✅ Supabase client initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            return None
            
    return _client
