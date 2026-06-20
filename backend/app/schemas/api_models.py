from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional

class IngestRequest(BaseModel):
    videoUrl: HttpUrl
    session_id: Optional[str] = None # Ties ingestion to a specific workspace/thread

class IngestResponse(BaseModel):
    status: str
    commentsProcessed: int
    message: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str # Ties chat to a specific workspace/thread
    message: str
    video_id: Optional[str] = None # Optional filter for isolated context
    history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    reply: str
    metadata: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    code: str
    message: str
    
class ErrorResponseWrapper(BaseModel):
    error: ErrorResponse
