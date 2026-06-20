from fastapi import APIRouter, Depends, HTTPException, Header, Query
from sqlalchemy.orm import Session
from app.database.connection import get_db
from app.database.models import Session as DbSession, Message, SessionVideo, YouTubeComment
from sqlalchemy import func
from pydantic import BaseModel
import uuid

router = APIRouter()

class CreateSessionRequest(BaseModel):
    user_id: str
    title: str = "New Workspace"

@router.post("/sessions")
def create_session(body: CreateSessionRequest, db: Session = Depends(get_db)):
    session_id = str(uuid.uuid4())
    new_session = DbSession(
        id=session_id,
        user_id=body.user_id,
        title=body.title
    )
    db.add(new_session)
    db.commit()
    return {"id": session_id, "title": new_session.title}

@router.get("/sessions")
def get_sessions(user_id: str, db: Session = Depends(get_db)):
    if user_id == "demouser":
        # Automatically migrate legacy anonymous sessions ('user-*') to 'demouser'
        db.query(DbSession).filter(DbSession.user_id.like("user-%")).update(
            {DbSession.user_id: "demouser"}, 
            synchronize_session=False
        )
        db.commit()

    sessions = db.query(DbSession).filter(DbSession.user_id == user_id).order_by(DbSession.created_at.desc()).all()
    return [{"id": s.id, "title": s.title, "created_at": s.created_at} for s in sessions]

@router.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str, db: Session = Depends(get_db)):
    messages = db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at.asc()).all()
    return [{"role": m.role, "content": m.content, "created_at": m.created_at} for m in messages]

@router.get("/sessions/{session_id}/videos")
def get_session_videos(session_id: str, db: Session = Depends(get_db)):
    videos = db.query(SessionVideo).filter(SessionVideo.session_id == session_id).order_by(SessionVideo.added_at.asc()).all()
    return [{"video_id": v.video_id, "added_at": v.added_at} for v in videos]

@router.get("/sessions/{session_id}/stats")
def get_session_stats(session_id: str, video_id: str = Query(None), db: Session = Depends(get_db)):
    # Aggregate sentiment_label counts for all videos linked to this session
    query = db.query(
        YouTubeComment.sentiment_label,
        func.count(YouTubeComment.id)
    ).join(
        SessionVideo, SessionVideo.video_id == YouTubeComment.video_id
    ).filter(
        SessionVideo.session_id == session_id
    )
    
    if video_id:
        query = query.filter(SessionVideo.video_id == video_id)
        
    stats = query.group_by(
        YouTubeComment.sentiment_label
    ).all()
    
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for label, count in stats:
        if label:
            sentiment_counts[label.upper()] = count
            
    return {"sentimentCounts": sentiment_counts}

class UpdateSessionRequest(BaseModel):
    title: str

@router.put("/sessions/{session_id}")
def update_session(session_id: str, body: UpdateSessionRequest, db: Session = Depends(get_db)):
    session = db.query(DbSession).filter(DbSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session.title = body.title
    db.commit()
    
    return {"status": "success", "title": session.title}

@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(DbSession).filter(DbSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # Delete associated messages
    db.query(Message).filter(Message.session_id == session_id).delete()
    # Delete associated videos mapping
    db.query(SessionVideo).filter(SessionVideo.session_id == session_id).delete()
    # Delete session
    db.delete(session)
    db.commit()
    
    return {"status": "success", "message": "Session deleted"}
