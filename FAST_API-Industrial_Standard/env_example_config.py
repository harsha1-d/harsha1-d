from pydantic import BaseSettings, Field

class Setting(BaseSettings):
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    database_url: str = Field(..., env="DATABASE_URL")
    debug: bool = Field(False, env="APP_DEBUG")
    
    class Config:
        env_file = ".env.example"  # Specify the path to your .env file
        
setting = Setting()
