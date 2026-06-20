import json
import os
from dotenv import load_dotenv

# Load .env so YouTube API key works
load_dotenv()

from app.services.ml_pipeline import sentiment_model
from app.services.youtube_api import fetch_top_comments

def test_ml_filter_with_youtube():
    print("Initializing ML Model Test with LIVE YouTube Data...\n")
    
    # Let's use the Project Zomboid video you tested earlier
    test_video_url = "https://www.youtube.com/watch?v=HS7wS4juHfs"
    print(f"--- 1. Fetching Top 5 Comments from {test_video_url} ---")
    
    try:
        comments_data = fetch_top_comments(test_video_url, max_results=5)
    except Exception as e:
        print(f"Failed to fetch YouTube comments: {e}")
        return

    # Extract just the text for ML processing
    test_comments = [c["text_display"] for c in comments_data]
    
    for i, text in enumerate(test_comments):
        print(f"[{i}] {text}")
        
    print("\n--- 2. Translating to English ---")
    translated_texts = sentiment_model.translate_batch(test_comments)
    for i, text in enumerate(translated_texts):
        print(f"[{i}] {text}")
        
    print("\n--- 3. Running ML Sentiment Prediction ---")
    sentiments = sentiment_model.predict(translated_texts)
    scores = sentiment_model.predict_proba(translated_texts)
    
    print("\n--- Final ML Output ---")
    results = []
    for i in range(len(test_comments)):
        results.append({
            "author": comments_data[i]["author"],
            "original_text": test_comments[i],
            "translated_text": translated_texts[i],
            "sentiment_label": sentiments[i],
            "confidence_score": round(scores[i], 3)
        })
        
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_ml_filter_with_youtube()
