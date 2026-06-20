import os
import json
from dotenv import load_dotenv

# We must load the .env file BEFORE importing the youtube_api
# so that os.getenv("YOUTUBE_API_KEY") catches the key during the import.
load_dotenv()

from app.services.youtube_api import fetch_top_comments

if __name__ == "__main__":
    # Using the user provided YouTube video
    test_video_url = "https://www.youtube.com/watch?v=I7qch6FMfoo"
    
    print(f"Fetching up to 5 comments for: {test_video_url}")
    print("-" * 50)
    
    try:
        # We limit to 5 results just so it doesn't flood your screen
        comments = fetch_top_comments(test_video_url, max_results=5)
        
        # Print the result nicely formatted
        print(json.dumps(comments, indent=2, ensure_ascii=True))
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Please make sure your YOUTUBE_API_KEY is valid and set in the backend/.env file.")
