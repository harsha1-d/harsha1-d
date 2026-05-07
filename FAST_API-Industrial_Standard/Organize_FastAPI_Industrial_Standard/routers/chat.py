from fastapi import APIRouter, HTTPException
from Organize_FastAPI_Industrial_Standard.models.chat import ChatRequest, ChatResponse
from Organize_FastAPI_Industrial_Standard.services.grok_services import get_groq_reply

router = APIRouter(prefix="/chat", tags=["Chatbot"])

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        reply = await get_groq_reply(request.user_message)
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))