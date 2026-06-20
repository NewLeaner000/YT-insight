from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from app.schemas.api_models import IngestRequest
from app.database.connection import get_db
from app.database.models import YouTubeComment, SessionVideo, Session as DbSession
from app.services.youtube_api import fetch_top_comments, fetch_video_title
from app.services.ml_pipeline import sentiment_model
from app.services.embeddings import embeddings_manager
import os

router = APIRouter()

@router.post("/ingest")
def ingest_video(request: Request, body: IngestRequest, db: Session = Depends(get_db)):
    # 1. Fetch Comments
    try:
        # Limit to 50 comments to prevent hitting the Gemini 100 requests/min rate limit
        comments_data = fetch_top_comments(str(body.videoUrl), max_results=50)
    except Exception as e:
        print(f"Error fetching comments: {e}")
        raise HTTPException(status_code=400, detail="Failed to fetch comments. Please check if the YouTube URL is valid and the video has comments enabled.")
        
    if not comments_data:
        raise HTTPException(status_code=404, detail="No comments found or video unavailable.")

    # 2. Translation Bridge & ML Sentiment Analysis
    original_texts = [c["text_display"] for c in comments_data]
    
    # Translate to English for ML inference
    translated_texts = sentiment_model.translate_batch(original_texts)
    
    # Run ML on English text
    sentiments = sentiment_model.predict(translated_texts)
    scores = sentiment_model.predict_proba(translated_texts)
    
    # 3. Vector Embeddings (Gemini)
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing.")
        
    embeddings_model = embeddings_manager.get_model()
    if not embeddings_model:
        raise HTTPException(status_code=500, detail="Embedding model is not configured properly.")
    
    try:
        # Embed the translated texts (English is better for embeddings)
        vectors = embeddings_model.embed_documents(translated_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

    video_id_extracted = comments_data[0]["video_id"] if comments_data else None

    # 4. Store in Neon Postgres (pgvector)
    try:
        # Check if video already ingested somewhat (Optional optimization, skipping for now)
        for i, c in enumerate(comments_data):
            db_comment = YouTubeComment(
                video_id=c["video_id"],
                author=c["author"],
                text_display=original_texts[i],
                translated_text=translated_texts[i],
                like_count=c["like_count"],
                published_at=c["published_at"],
                sentiment_label=sentiments[i],
                sentiment_score=scores[i],
                embedding=vectors[i]
            )
            db.add(db_comment)
            
        # 5. Link video to session if session_id is provided
        video_title = video_id_extracted
        if body.session_id and video_id_extracted:
            # Check if link already exists to avoid duplicates
            existing_link = db.query(SessionVideo).filter(
                SessionVideo.session_id == body.session_id,
                SessionVideo.video_id == video_id_extracted
            ).first()
            if not existing_link:
                new_link = SessionVideo(session_id=body.session_id, video_id=video_id_extracted)
                db.add(new_link)
            
            # Update session title if it's currently "New Workspace"
            current_session = db.query(DbSession).filter(DbSession.id == body.session_id).first()
            if current_session and current_session.title == "New Workspace":
                video_title = fetch_video_title(str(body.videoUrl))
                current_session.title = video_title
                db.add(current_session)
                
        db.commit()
    except Exception as e:
        db.rollback()
        safe_e = str(e).encode('utf-8', 'replace').decode('utf-8')
        print(f"Database insertion failed: {safe_e}")
        raise HTTPException(status_code=500, detail=f"Database error: {safe_e}")
        
    # Convert sentiments to uppercase to avoid case-mismatch bugs
    sentiments_upper = [str(s).upper() for s in sentiments]
    
    # Calculate Sentiment Distribution
    sentiment_counts = {
        "POSITIVE": sentiments_upper.count("POSITIVE"),
        "NEGATIVE": sentiments_upper.count("NEGATIVE"),
        "NEUTRAL": sentiments_upper.count("NEUTRAL")
    }

    return {
        "status": "success",
        "commentsProcessed": len(comments_data),
        "message": f"Successfully ingested {len(comments_data)} comments into the database.",
        "videoId": video_id_extracted,
        "videoTitle": video_title if 'video_title' in locals() else video_id_extracted,
        "videoUrl": str(body.videoUrl),
        "sentimentCounts": sentiment_counts
    }
