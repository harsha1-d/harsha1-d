from fastapi import FastAPI
from Organize_FastAPI_Industrial_Standard.config import settings
from Organize_FastAPI_Industrial_Standard.routers import chat

app = FastAPI(title=settings.app_name, version="1.0.0")

app.include_router(chat.router)
