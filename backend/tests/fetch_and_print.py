import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from app.services.youtube_api import fetch_top_comments

print("Fetching comments for Q-K0BdKbaNQ...")
try:
    comments = fetch_top_comments("Q-K0BdKbaNQ", max_results=15)
    with open("golden_comments.txt", "w", encoding="utf-8") as f:
        for c in comments:
            f.write(f"- {c['text_display']}\n")
    print("Saved to golden_comments.txt")
except Exception as e:
    print(f"Error: {e}")
