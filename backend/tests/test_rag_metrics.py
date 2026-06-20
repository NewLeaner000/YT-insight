import pytest
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from app.agent.tools import get_all_agent_tools
from app.agent.bot import SYSTEM_PROMPT

# We omit mistral from this specific RAG test because it already 
# failed the basic tool routing compilation test in test_agent.py
TEST_MODELS = ["llama3.1", "qwen2.5"] 

@pytest.fixture(scope="module")
def tools():
    return get_all_agent_tools()

@pytest.mark.parametrize("model_name", TEST_MODELS)
def test_rag_context_and_faithfulness(model_name, tools):
    """
    Test Context Precision, Faithfulness, and Speed based on the Golden Dataset.
    Question: 'What do the YouTube comments say about Endmin's mask?'
    Expected: Retrieve relevant comments from DB and correctly identify users want an unmasked character.
    """
    try:
        llm = ChatOllama(model=model_name, temperature=0)
        agent = create_react_agent(llm, tools, prompt=SystemMessage(content=SYSTEM_PROMPT))
        
        start_time = time.time()
        response = agent.invoke({"messages": [HumanMessage(content="What do the YouTube comments say about Endmin's mask?")]})
        latency = time.time() - start_time
        
        # 1. Latency/Speed Assertion
        assert latency < 35.0, f"[{model_name}] Speed test failed: Took {latency:.2f}s (Threshold is 35s)"
        
        # Extract tools
        tool_calls = []
        for msg in response["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for call in msg.tool_calls:
                    tool_calls.append(call['name'])
                    
        # 2. Context Precision Assertion (Did it use Semantic Search for a contextual question?)
        assert "Semantic_Comment_Search" in tool_calls, f"[{model_name}] Failed to use Semantic Search for a qualitative question."
        
        # 3. Faithfulness Assertion (Did it answer correctly based on the Golden Dataset?)
        final_message = response["messages"][-1].content
        if isinstance(final_message, list):
            final_answer = " ".join([b.get("text", "") for b in final_message if isinstance(b, dict)]).lower()
        else:
            final_answer = str(final_message).lower()
            
        # The Golden Dataset comments explicitly mention wanting "unmasked endmind" and "Endmin without Mask".
        assert "unmask" in final_answer or "without mask" in final_answer or "remove" in final_answer, \
            f"[{model_name}] Faithfulness failed! Model hallucinated or missed the detail. Final Output: '{final_answer}'"
            
    except Exception as e:
        pytest.fail(f"[{model_name}] Raised an exception during testing: {e}")
