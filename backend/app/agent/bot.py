import google.generativeai as genai
import google.ai.generativelanguage as glm
from google.protobuf.struct_pb2 import Struct
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

MODEL_FALLBACK_CHAIN = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

_agent_cache = {}
_exhausted_models = set()


def _convert_tools_to_genai(langchain_tools):
    """Convert langchain Tool objects to Google GenAI tool config (plain dicts)."""
    fn_decls = []
    for tool in langchain_tools:
        fn_decls.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "input": {
                        "type_": "STRING",
                        "description": "The input query or parameter for this tool"
                    }
                },
                "required": ["input"]
            }
        })
    return fn_decls


def _run_agent_loop(model, tool_map, system_message, chat_history_messages):
    """Execute the ReAct agent loop using Google GenAI SDK."""

    # Build history
    history = []
    for msg in chat_history_messages:
        if msg["role"] == "user":
            history.append(glm.Content(
                role="user",
                parts=[glm.Part(text=msg["content"])]
            ))
        elif msg["role"] == "ai":
            history.append(glm.Content(
                role="model",
                parts=[glm.Part(text=msg["content"])]
            ))

    # Separate last user message
    if history:
        last_content = history.pop()
        first_text = last_content.parts[0].text
    else:
        first_text = "Hello"

    chat = model.start_chat(history=history)

    # Prepend system prompt to first message (0.4.x has no system_instruction)
    if not history:
        current_message = f"[SYSTEM INSTRUCTIONS]\n{system_message}\n\n[USER QUESTION]\n{first_text}"
    else:
        current_message = first_text

    # ReAct loop — max 10 iterations
    for _ in range(10):
        response = chat.send_message(current_message)

        # Check for function calls
        function_calls = []
        for part in response.parts:
            if part.function_call and part.function_call.name:
                function_calls.append(part.function_call)

        if not function_calls:
            # No tool calls — return text
            return response.text

        # Execute function calls
        fn_response_parts = []
        for fc in function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args) if fc.args else {}

            if tool_name in tool_map:
                tool = tool_map[tool_name]
                tool_input = (
                    tool_args.get("input", "") or
                    tool_args.get("query", "") or
                    tool_args.get("sentiment", "") or
                    tool_args.get("sql_query", "")
                )
                if not tool_input and tool_args:
                    tool_input = str(list(tool_args.values())[0])

                try:
                    result = tool.func(tool_input)
                except Exception as e:
                    result = f"Tool error: {str(e)}"
            else:
                result = f"Unknown tool: {tool_name}"

            # Build function response using glm types
            s = Struct()
            s.update({"result": result})
            fn_response_parts.append(
                glm.Part(function_response=glm.FunctionResponse(
                    name=tool_name, response=s
                ))
            )

        # Send function results back to model
        current_message = glm.Content(parts=fn_response_parts)

    return "Unable to find a conclusive answer. Please try rephrasing your question."


def _create_agent(model_name: str, video_ids: list = None):
    """Create agent using Google GenAI SDK directly."""
    print(f"[Agent] Creating agent with model: {model_name}")

    from app.agent.tools import get_all_agent_tools
    langchain_tools = get_all_agent_tools(video_ids=video_ids)

    tool_map = {tool.name: tool for tool in langchain_tools}
    fn_decls = _convert_tools_to_genai(langchain_tools)

    system_message = SYSTEM_PROMPT
    if video_ids:
        v_list = "', '".join(video_ids)
        system_message += f"\nCRITICAL INSTRUCTION: You are answering questions EXCLUSIVELY about the YouTube Videos with IDs: '{v_list}'. When using Run_ReadOnly_SQL, you MUST always include `WHERE video_id IN ('{v_list}')` in your query."

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Use plain dict for tools config — works with ALL SDK versions
    model = genai.GenerativeModel(
        model_name=model_name,
        tools=[{"function_declarations": fn_decls}],
        generation_config={"temperature": 0},
    )

    return model, tool_map, system_message


def get_agent_executor(video_ids: list = None, model_name: str = None):
    """Returns (model, tool_map, system_message) tuple."""
    if not os.getenv("DATABASE_URL") or not os.getenv("GOOGLE_API_KEY"):
        return None

    vid_key = ",".join(sorted(video_ids)) if video_ids else "__global__"

    if model_name:
        cache_key = f"{model_name}:{vid_key}"
        if cache_key not in _agent_cache:
            _agent_cache[cache_key] = _create_agent(model_name, video_ids)
        return _agent_cache[cache_key]

    for m in MODEL_FALLBACK_CHAIN:
        if m not in _exhausted_models:
            cache_key = f"{m}:{vid_key}"
            if cache_key not in _agent_cache:
                _agent_cache[cache_key] = _create_agent(m, video_ids)
            return _agent_cache[cache_key]

    return None


def mark_model_exhausted(model_name: str):
    _exhausted_models.add(model_name)
    print(f"[Agent] Model '{model_name}' marked as EXHAUSTED. Remaining: "
          f"{[m for m in MODEL_FALLBACK_CHAIN if m not in _exhausted_models]}")


def get_available_models():
    return [m for m in MODEL_FALLBACK_CHAIN if m not in _exhausted_models]
