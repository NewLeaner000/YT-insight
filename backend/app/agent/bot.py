import google.generativeai as genai
import os
import json

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


def _convert_tools_to_genai(langchain_tools):
    """Convert langchain Tool objects to Google GenAI function declarations."""
    function_declarations = []
    for tool in langchain_tools:
        func_decl = {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The input query or parameter for this tool"
                    }
                },
                "required": ["input"]
            }
        }
        function_declarations.append(func_decl)
    return function_declarations


def _run_agent_loop(model, tool_map, system_message, chat_history_messages):
    """Execute the ReAct agent loop: LLM → tool call → LLM → ... → final answer."""
    
    # Build conversation history for Google GenAI format
    history = []
    for msg in chat_history_messages:
        if msg["role"] == "user":
            history.append({"role": "user", "parts": [msg["content"]]})
        elif msg["role"] == "ai":
            history.append({"role": "model", "parts": [msg["content"]]})
    
    chat = model.start_chat(history=history[:-1] if history else [])
    
    # Send the last user message (or empty if no history)
    last_message = history[-1]["parts"][0] if history else "Hello"
    
    # ReAct loop — max 10 iterations to prevent infinite loops
    for _ in range(10):
        response = chat.send_message(last_message)
        
        # Check if the model wants to call a function
        function_calls = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call.name:
                function_calls.append(part.function_call)
        
        if not function_calls:
            # No function calls — return the text response
            return response.text
        
        # Execute each function call
        function_responses = []
        for fc in function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args) if fc.args else {}
            
            if tool_name in tool_map:
                tool = tool_map[tool_name]
                # Extract the input argument
                tool_input = tool_args.get("input", tool_args.get("query", tool_args.get("sentiment", tool_args.get("sql_query", ""))))
                
                # If no standard key found, try the first available value
                if not tool_input and tool_args:
                    tool_input = str(list(tool_args.values())[0])
                
                try:
                    result = tool.func(tool_input)
                except Exception as e:
                    result = f"Tool error: {str(e)}"
            else:
                result = f"Unknown tool: {tool_name}"
            
            function_responses.append(
                genai.protos.Part(function_response=genai.protos.FunctionResponse(
                    name=tool_name,
                    response={"result": result}
                ))
            )
        
        # Send function results back to the model
        last_message = function_responses
    
    return "I was unable to find a conclusive answer after multiple attempts. Please try rephrasing your question."


def _create_agent(model_name: str, video_ids: list = None):
    """Create agent using Google GenAI SDK directly — no langgraph needed."""
    print(f"[Agent] Creating agent with model: {model_name}")
    
    from app.agent.tools import get_all_agent_tools
    langchain_tools = get_all_agent_tools(video_ids=video_ids)
    
    # Build tool map for execution
    tool_map = {tool.name: tool for tool in langchain_tools}
    
    # Convert to GenAI function declarations
    function_declarations = _convert_tools_to_genai(langchain_tools)
    
    # Build system message
    system_message = SYSTEM_PROMPT
    if video_ids:
        v_list = "', '".join(video_ids)
        system_message += f"\nCRITICAL INSTRUCTION: You are answering questions EXCLUSIVELY about the YouTube Videos with IDs: '{v_list}'. When using Run_ReadOnly_SQL, you MUST always include `WHERE video_id IN ('{v_list}')` in your query."
    
    # Configure API key
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Create model with tools
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_message,
        tools=[genai.protos.Tool(function_declarations=[
            genai.protos.FunctionDeclaration(
                name=fd["name"],
                description=fd["description"],
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        k: genai.protos.Schema(type=genai.protos.Type.STRING, description=v.get("description", ""))
                        for k, v in fd["parameters"].get("properties", {}).items()
                    },
                    required=fd["parameters"].get("required", [])
                )
            )
            for fd in function_declarations
        ])]
    )
    
    return model, tool_map, system_message


def get_agent_executor(video_ids: list = None, model_name: str = None):
    """
    Returns (model, tool_map, system_message) tuple.
    Uses model fallback chain if no specific model is given.
    """
    if not os.getenv("DATABASE_URL") or not os.getenv("GOOGLE_API_KEY"):
        return None
    
    vid_key = ",".join(sorted(video_ids)) if video_ids else "__global__"
    
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
    
    return None


def mark_model_exhausted(model_name: str):
    """Mark a model as exhausted (daily quota hit)."""
    _exhausted_models.add(model_name)
    print(f"[Agent] Model '{model_name}' marked as EXHAUSTED. Remaining models: "
          f"{[m for m in MODEL_FALLBACK_CHAIN if m not in _exhausted_models]}")


def get_available_models():
    """Returns list of models that haven't been exhausted yet."""
    return [m for m in MODEL_FALLBACK_CHAIN if m not in _exhausted_models]
