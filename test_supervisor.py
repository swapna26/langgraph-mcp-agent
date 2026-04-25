"""Quick test to debug the supervisor routing."""

import os
import asyncio
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.supervisor import SupervisorOutput

load_dotenv()
logging.basicConfig(level=logging.INFO)


async def test():
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
    )

    system_prompt = """You are a supervisor routing user queries to specialized agents.

Available agents:
- web_rag_agent: Handles web search questions
- doc_rag_agent: Handles questions about internal documents (HR, Procurement, Security)
- db_rag_agent: Handles database/SQL queries

Route the user query to the appropriate agent(s). You MUST choose at least one agent."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What are the procurement rules in Abu Dhabi?"},
    ]

    print("Test 1: Structured output...")
    try:
        response = llm.with_structured_output(SupervisorOutput).invoke(messages)
        print(f"  next_agents: {response.next_agents}")
        print(f"  modified_queries: {response.modified_queries}")
        print(f"  reasoning: {response.reasoning}")
        print("  ✓ PASSED\n")
    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")

    print("Test 2: Raw invoke...")
    try:
        response = llm.invoke("Say hello in 3 words")
        print(f"  Response: {response.content}")
        print("  ✓ PASSED\n")
    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")


asyncio.run(test())
