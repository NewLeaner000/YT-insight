import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from app.database.connection import SessionLocal
from app.database.models import YouTubeComment
from app.services.youtube_api import fetch_top_comments
from app.services.ml_pipeline import sentiment_model
from app.services.embeddings import embeddings_manager

def seed_golden():
    video_id = "Q-K0BdKbaNQ"
    golden_video_id = "GOLDEN_TEST_01"
    
    print(f"Fetching comments for {video_id}...")
    comments_data = fetch_top_comments(video_id, max_results=15)
    
    for c in comments_data:
        c["video_id"] = golden_video_id
        
    print(f"Fetched {len(comments_data)} comments. Running ML Translation, Sentiment, and Embeddings...")
    original_texts = [c["text_display"] for c in comments_data]
    
    # Pre-process using the live ML pipeline
    translated_texts = sentiment_model.translate_batch(original_texts)
    sentiments = sentiment_model.predict(translated_texts)
    scores = sentiment_model.predict_proba(translated_texts)
    
    embeddings_model = embeddings_manager.get_model()
    vectors = embeddings_model.embed_documents(translated_texts)
    
    print(f"Saving to database under video_id: {golden_video_id}...")
    db = SessionLocal()
    try:
        # Wipe old test data to keep it idempotent
        db.query(YouTubeComment).filter(YouTubeComment.video_id == golden_video_id).delete()
        
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
        db.commit()
        print("Golden Dataset successfully seeded!")
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_golden()
