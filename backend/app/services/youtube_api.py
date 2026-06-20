import os
from googleapiclient.discovery import build
from typing import List, Dict

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

import re

def extract_video_id(url: str) -> str:
    video_id = url
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    elif "/shorts/" in url:
        video_id = url.split("/shorts/")[1].split("?")[0]
        
    # Validate the 11-character YouTube ID format
    if not re.match(r"^[A-Za-z0-9_-]{11}$", video_id):
        raise ValueError("Invalid YouTube URL or Video ID format.")
        
    return video_id

def fetch_video_title(video_url: str) -> str:
    if not YOUTUBE_API_KEY:
        return "Unknown Video"
    try:
        video_id = extract_video_id(video_url)
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        if response.get("items"):
            return response["items"][0]["snippet"]["title"]
    except Exception as e:
        print(f"Error fetching video title: {e}")
    return extract_video_id(video_url)

def fetch_top_comments(video_url: str, max_results: int = 200) -> List[Dict]:
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY is not set. Please add it to your .env file.")
    
    video_id = extract_video_id(video_url)
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,  # Max per page is 100
        textFormat="plainText"
    )
    
    while request and len(comments) < max_results:
        response = request.execute()
        
        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "author": snippet["authorDisplayName"],
                "text_display": snippet["textDisplay"],
                "like_count": snippet["likeCount"],
                "published_at": snippet["publishedAt"]
            })
            if len(comments) >= max_results:
                break
                
        request = youtube.commentThreads().list_next(request, response)
        
    return comments
