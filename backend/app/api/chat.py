import time
import re
from fastapi import APIRouter, Depends, Request, HTTPException
from app.schemas.api_models import ChatRequest
from app.agent.bot import get_agent_executor, mark_model_exhausted, get_available_models, MODEL_FALLBACK_CHAIN, _run_agent_loop

from sqlalchemy.orm import Session
from app.database.connection import get_db
from app.database.models import Message, SessionVideo

router = APIRouter()


def _extract_model_from_error(error_str: str) -> str:
    """Extract the model name from a 429 error message."""
    match = re.search(r'model:\s*(gemini-[\w.-]+)', error_str)
    return match.group(1) if match else None


@router.post("/chat")
def chat_with_agent(request: Request, body: ChatRequest, db: Session = Depends(get_db)):
    session_id = body.session_id
    
    # Fetch all video_ids associated with this session
    session_videos = db.query(SessionVideo).filter(SessionVideo.session_id == session_id).all()
    video_ids = [sv.video_id for sv in session_videos]
    
    # If client requested context isolation for a specific video
    if body.video_id and body.video_id in video_ids:
        video_ids = [body.video_id]
    
    # Save user message to DB
    user_msg = Message(session_id=session_id, role="user", content=body.message)
    db.add(user_msg)
    db.commit()
    
    # Build chat history in simple dict format
    chat_history = []
    db_history = db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at.asc()).all()
    for msg in db_history:
        if msg.role == "user":
            chat_history.append({"role": "user", "content": msg.content})
        elif msg.role == "ai":
            chat_history.append({"role": "ai", "content": msg.content})
    
    # Try each model in the fallback chain
    for model_name in MODEL_FALLBACK_CHAIN:
        available = get_available_models()
        if model_name not in available:
            continue
        
        result = get_agent_executor(video_ids=video_ids, model_name=model_name)
        if not result:
            continue
        model, tool_map, system_message = result
        
        try:
            print(f"[Chat] Trying model: {model_name}")
            final_message = _run_agent_loop(model, tool_map, system_message, chat_history)
            
            # Ensure it's a string
            if not isinstance(final_message, str):
                final_message = str(final_message)
                
            # Save AI response to DB
            ai_msg = Message(session_id=session_id, role="ai", content=final_message)
            db.add(ai_msg)
            db.commit()
            
            return {
                "reply": final_message,
                "metadata": {
                    "model": model_name,
                    "toolUsed": "Google GenAI ReAct Agent (Hybrid)"
                }
            }
        except Exception as e:
            error_str = str(e)
            
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Mark this model as exhausted and try the next one
                mark_model_exhausted(model_name)
                print(f"[Chat] Model '{model_name}' quota exhausted. Falling back...")
                continue
            else:
                raise HTTPException(status_code=500, detail=error_str)
    
    # All models exhausted
    raise HTTPException(
        status_code=429,
        detail="All Gemini models have reached their daily free tier quota. Please wait until tomorrow or upgrade your API key to a paid plan."
    )
