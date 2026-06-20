import os
import sys
import time
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from app.agent.tools import get_all_agent_tools
from app.agent.bot import SYSTEM_PROMPT

def evaluate_numbers():
    print("Initializing RAG LLM-Judge Evaluation for Gemini-2.5-Flash...\n")
    tools = get_all_agent_tools()
    
    # Initialize the LLM Judge (Gemini)
    try:
        judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        # Agent uses the same
        agent_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        return
    
    # Print Table Header
    print(f"{'Model':<20} | {'Latency (sec)':<15} | {'Context Hit Rate (%)':<25} | {'Faithfulness Score (0-100)':<25}")
    print("-" * 95)
    
    try:
        agent = create_react_agent(agent_llm, tools, prompt=SystemMessage(content=SYSTEM_PROMPT))
        
        start_time = time.time()
        response = agent.invoke({"messages": [HumanMessage(content="What do the YouTube comments say about Endmin's mask?")]})
        latency = time.time() - start_time
        
        # Extract Context and Answer
        context_text = "No context retrieved."
        tool_messages = [m for m in response["messages"] if m.type == "tool"]
        if tool_messages:
            context_text = tool_messages[0].content
            
        final_message = response["messages"][-1].content
        if isinstance(final_message, list):
            final_answer = " ".join([b.get("text", "") for b in final_message if isinstance(b, dict)])
        else:
            final_answer = str(final_message)
            
        # Calculate Context Hit Rate
        expected_keywords = ["unmasked", "without mask", "zhuang"]
        hits = sum(1 for k in expected_keywords if k in context_text.lower())
        context_hit_rate = (hits / len(expected_keywords)) * 100.0 if expected_keywords else 0
        
        # Use Gemini as the LLM Judge for Faithfulness
        judge_prompt = f"""
        You are an impartial and strict judge.
        
        Here is the Context retrieved from the database:
        {context_text}
        
        Here is the local LLM's Answer based on the Context:
        {final_answer}
        
        Evaluate the LLM's Answer based ONLY on the Context. Does the answer faithfully represent the information in the context? Does it mention that people want to see Endmin without the mask? Did it hallucinate any information not present in the context?
        
        Reply with ONLY a single integer score from 0 to 100 representing the faithfulness. DO NOT provide any other text.
        """
        
        judge_response = judge_llm.invoke([HumanMessage(content=judge_prompt)])
        
        # Extract the integer score safely
        match = re.search(r'\d+', judge_response.content)
        if match:
            faithfulness_score = int(match.group())
        else:
            faithfulness_score = 0
            
        # Print Row
        print(f"{'gemini-2.5-flash':<20} | {latency:<15.2f} | {context_hit_rate:<25.1f} | {faithfulness_score:<25}")
        
    except Exception as e:
        print(f"ERROR during evaluation: {e}")

if __name__ == "__main__":
    evaluate_numbers()
