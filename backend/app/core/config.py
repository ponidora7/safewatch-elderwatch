from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "SafeWatch API"
    API_V1_STR: str = "/api/v1"
    
    # Supabase
    SUPABASE_URL: str = "https://placeholder-project.supabase.co"
    SUPABASE_KEY: str = "placeholder-key"
    
    # AI Inference (Hugging Face)
    HF_API_URL: str = "https://api-inference.huggingface.co/models/placeholder"
    HF_TOKEN: Optional[str] = None
    CONFIDENCE_THRESHOLD: float = 0.7
    
    # Email (Resend)
    RESEND_API_KEY: str = "re_placeholder"
    EMAIL_FROM: str = "onboarding@resend.dev"
    EMAIL_TO: str = "recipient@example.com"
    
    # Cooldown logic (minutes)
    NOTIFICATION_COOLDOWN: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
