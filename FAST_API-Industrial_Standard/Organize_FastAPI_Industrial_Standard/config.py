from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Groq Chatbot API"
    groq_api_key: str
    model: str = "llama-3.3-70b-versatile"
    
    class Config:
        env_file = ".env"
        
settings = Settings()