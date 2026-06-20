import time
import re
from fastapi import APIRouter, Depends, Request, HTTPException
from app.schemas.api_models import ChatRequest
from app.agent.bot import get_agent_executor, mark_model_exhausted, get_available_models, MODEL_FALLBACK_CHAIN
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
    
    # Format chat history for LangGraph
    chat_history = []
    
    # Always fetch history from DB to ensure consistency
    db_history = db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at.asc()).all()
    for msg in db_history:
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "ai":
            chat_history.append(AIMessage(content=msg.content))
    
    # Note: user_msg is already in db_history because we committed it above
    
    # Try each model in the fallback chain
    for model_name in MODEL_FALLBACK_CHAIN:
        available = get_available_models()
        if model_name not in available:
            continue
        
        result = get_agent_executor(video_ids=video_ids, model_name=model_name)
        if not result:
            continue
        agent_executor, system_message = result
        
        try:
            print(f"[Chat] Trying model: {model_name}")
            # Prepend SystemMessage — compatible with ALL langgraph versions
            messages_with_system = [SystemMessage(content=system_message)] + chat_history
            response = agent_executor.invoke({"messages": messages_with_system})
            
            final_message = response["messages"][-1].content
            
            # Ensure it's a string
            if isinstance(final_message, list):
                text_blocks = []
                for block in final_message:
                    if isinstance(block, dict) and "text" in block:
                        text_blocks.append(block["text"])
                    elif isinstance(block, str):
                        text_blocks.append(block)
                final_message = " ".join(text_blocks)
            elif not isinstance(final_message, str):
                final_message = str(final_message)
                
            # Save AI response to DB
            ai_msg = Message(session_id=session_id, role="ai", content=final_message)
            db.add(ai_msg)
            db.commit()
            
            return {
                "reply": final_message,
                "metadata": {
                    "model": model_name,
                    "toolUsed": "LangGraph ReAct Agent (Hybrid)"
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
