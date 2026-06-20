from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.database.connection import Base

class YouTubeComment(Base):
    __tablename__ = "youtube_comments"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String(50), index=True, nullable=False)
    author = Column(String(100))
    text_display = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=True)
    like_count = Column(Integer, default=0)
    published_at = Column(DateTime(timezone=True), default=func.now())
    
    # Scikit-learn sentiment scores
    sentiment_label = Column(String(20)) # POSITIVE, NEGATIVE, NEUTRAL
    sentiment_score = Column(Float)
    
    # Text Embedding (3072 dims for Gemini)
    embedding = Column(Vector(3072))

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(String(50), primary_key=True, index=True)
    user_id = Column(String(100), index=True, nullable=False) # Maps to Supabase Auth UUID
    title = Column(String(200), default="New Chat")
    created_at = Column(DateTime(timezone=True), default=func.now())

class SessionVideo(Base):
    __tablename__ = "session_videos"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), index=True, nullable=False)
    video_id = Column(String(50), index=True, nullable=False)
    added_at = Column(DateTime(timezone=True), default=func.now())

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), index=True, nullable=False)
    role = Column(String(20), nullable=False) # 'user', 'ai', 'system'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now())
