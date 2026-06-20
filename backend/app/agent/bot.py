from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from app.agent.tools import get_all_agent_tools
import os

SYSTEM_PROMPT = """You are the YouTube Live Insight Agent. You analyze YouTube comments.
You have access to FIVE tools. Choose the most efficient one:

1. **Get_Sentiment_Summary**: Returns count/percentage of POSITIVE, NEGATIVE, NEUTRAL comments. Use for "how many positive?", "sentiment ratio?".
2. **Get_Top_Comments**: Returns top comments for a given sentiment (POSITIVE/NEGATIVE/NEUTRAL). Use for "what did positive people say?", "show negative comments".
3. **Get_Comment_Stats**: Returns general stats (total, likes, top author, most liked, date range). Use for "how many comments?", "who comments the most?".
4. **Semantic_Comment_Search**: Vector search for qualitative questions. Use for "why are people unhappy?", "what do they say about X?".
5. **Run_ReadOnly_SQL**: LAST RESORT. Write a raw SELECT query only when tools 1-4 cannot answer. Table: `youtube_comments`. Columns: `id`, `video_id`, `author`, `text_display`, `translated_text`, `like_count`, `published_at`, `sentiment_label`, `sentiment_score`.

PRIORITY ORDER: Always try tools 1-4 first. Only use Run_ReadOnly_SQL if the question truly requires custom aggregation.

Instructions to PREVENT HALLUCINATION:
- ALWAYS rely ON THE EXACT DATA returned by tools to answer user questions.
- NEVER make up, guess, or infer comments, sentiments, or statistics not provided by the tools.
- If tools return no data, say "I don't have enough data to answer that."
- When quoting comments, stay entirely faithful to the tool output.

SECURITY CONSTRAINT:
- You are strictly a READ-ONLY agent.
- NEVER execute SQL that modifies data (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE).
- If a user asks to modify data, refuse immediately.
"""

# Model fallback chain: try each model in order until one works
MODEL_FALLBACK_CHAIN = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# Cache agent instances by (model, video_id)
_agent_cache = {}

# Track which models are exhausted (hit daily quota)
_exhausted_models = set()


def _create_agent(model_name: str, video_ids: list[str] = None):
    """Create a ReAct agent with a specific Gemini model."""
    print(f"[Agent] Creating agent with model: {model_name}")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    
    tools = get_all_agent_tools(video_ids=video_ids)
    
    system_message = SYSTEM_PROMPT
    if video_ids:
        v_list = "', '".join(video_ids)
        system_message += f"\nCRITICAL INSTRUCTION: You are answering questions EXCLUSIVELY about the YouTube Videos with IDs: '{v_list}'. When using Run_ReadOnly_SQL, you MUST always include `WHERE video_id IN ('{v_list}')` in your query."
    
    return create_react_agent(llm, tools, state_modifier=SystemMessage(content=system_message))


def get_agent_executor(video_ids: list[str] = None, model_name: str = None):
    """
    Returns an agent executor. Uses model fallback chain if no specific model is given.
    Skips models that are known to be exhausted.
    """
    if not os.getenv("DATABASE_URL") or not os.getenv("GOOGLE_API_KEY"):
        return None
    
    vid_key = ",".join(sorted(video_ids)) if video_ids else "__global__"
    
    # If a specific model is requested
    if model_name:
        cache_key = f"{model_name}:{vid_key}"
        if cache_key not in _agent_cache:
            _agent_cache[cache_key] = _create_agent(model_name, video_ids)
        return _agent_cache[cache_key]
    
    # Auto-select: pick the first non-exhausted model
    for model in MODEL_FALLBACK_CHAIN:
        if model not in _exhausted_models:
            cache_key = f"{model}:{vid_key}"
            if cache_key not in _agent_cache:
                _agent_cache[cache_key] = _create_agent(model, video_ids)
            return _agent_cache[cache_key]
    
    # All models exhausted
    return None


def mark_model_exhausted(model_name: str):
    """Mark a model as exhausted (daily quota hit)."""
    _exhausted_models.add(model_name)
    print(f"[Agent] Model '{model_name}' marked as EXHAUSTED. Remaining models: "
          f"{[m for m in MODEL_FALLBACK_CHAIN if m not in _exhausted_models]}")


def get_available_models():
    """Returns list of models that haven't been exhausted yet."""
    return [m for m in MODEL_FALLBACK_CHAIN if m not in _exhausted_models]
