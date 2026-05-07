import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

#load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")

app= FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    tone: str = "neutral" #default value is neutral-extra field for business logic
    
@app.post("/genai")
def generate_text(request: PromptRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing. Add it to your .env file.",
        )
    
    tone_prefix = {
        "friendly": "Respond in a casual and warm tone.",
        "formal": "Respond in a professional and concise tone.",
        "neutral": "Respond neutrally."
    }.get(request.tone.lower(), "Respond neutrally.")
    
    final_prompt = f"{tone_prefix}\n\nUser request: {request.prompt}"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": final_prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to Groq API: {exc}",
        ) from exc

    if response.status_code != 200:
        try:
            error_detail = response.json()
        except ValueError:
            error_detail = response.text

        raise HTTPException(status_code=response.status_code, detail=error_detail)

    try:
        data = response.json()
        generated_text = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Unexpected response format from Groq API.",
        ) from exc

    return {
        "tone_used": request.tone,
        "original_prompt": request.prompt,
        "final_prompt": final_prompt,
        "response": generated_text
    }
