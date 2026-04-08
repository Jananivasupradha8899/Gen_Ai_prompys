from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.callbacks.manager import get_openai_callback
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# ==========================================
# 1. State Definition
# ==========================================
class ResearchState(TypedDict):
    query: str
    search_results: str
    draft: str
    editor_feedback: str
    final_report: str
    approval_status: str  # "pending", "approved", "rejected"
    search_depth: int
    total_tokens: int
    total_cost: float
    logs: List[str]

# ==========================================
# 2. Agent Nodes
# ==========================================

def etl_preprocessor(state: ResearchState) -> Dict[str, Any]:
    """Cleans the query and prepares it for research."""
    query = state["query"].strip()
    logs = state.get("logs", [])
    logs.append(f"ETL Node: Preprocessed query -> '{query}'")
    return {
        "query": query, 
        "logs": logs, 
        "approval_status": "in_progress",
        "total_tokens": state.get("total_tokens", 0),
        "total_cost": state.get("total_cost", 0.0)
    }

def researcher_agent(state: ResearchState) -> Dict[str, Any]:
    """Searches the web using DuckDuckGo to gather real data."""
    query = state["query"]
    logs = state.get("logs", [])
    logs.append(f"Researcher Agent: Searching live web for '{query}'...")
    
    try:
        results = ""
        depth = state.get("search_depth", 2)
        with DDGS() as ddgs:
            # depth is configurable via UI
            search_gen = ddgs.text(query, max_results=depth, region='wt-wt', safesearch='off', timelimit='y')
            for r in search_gen:
                title = r.get('title', 'Unknown')
                body = r.get('body', '')
                href = r.get('href', 'No Link')
                results += f"Source: {title} ({href})\n{body}\n\n"
                
        if not results.strip():
            results = "No high-confidence web results found. Proceeding with LLM internal knowledge."
            
        logs.append(f"Researcher Agent: Successfully extracted search context.")
    except Exception as e:
        results = f"Web search timeout/error: {str(e)}. Proceeding with LLM internal knowledge."
        logs.append(f"Researcher Agent: {results}")

    return {"search_results": results, "logs": logs}

def writer_agent(state: ResearchState) -> Dict[str, Any]:
    """Uses OpenAI to synthesize the research into a draft."""
    query = state["query"]
    context = state["search_results"]
    logs = state.get("logs", [])
    
    logs.append("Writer Agent: Connecting to OpenAI API...")
    
    # Check for API Key
    if not os.environ.get("OPENAI_API_KEY"):
        error_msg = "Error: OPENAI_API_KEY is not set."
        logs.append("Writer Agent: " + error_msg)
        return {"draft": error_msg, "logs": logs}
    
    # Use gpt-4o-mini for much faster, cheaper generation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    prompt = f"""
    You are an expert technical writer. Write a concise, 2-3 paragraph informative report on the following topic:
    TOPIC: "{query}"
    
    Use the following LIVE SEARCH context to ground your writing (If context says 'search failed', rely on your internal knowledge):
    ---
    {context}
    ---
    
    Format nicely in Markdown.
    """
    
    try:
        with get_openai_callback() as cb:
            response = llm.invoke([SystemMessage(content="You are a skilled technical writer."), HumanMessage(content=prompt)])
            draft = response.content
            logs.append("Writer Agent: Draft generated successfully.")
            tokens = state.get("total_tokens", 0) + cb.total_tokens
            cost = state.get("total_cost", 0.0) + cb.total_cost
    except Exception as e:
        draft = f"API Error: {e}"
        logs.append(f"Writer Agent: Failed to generate draft. Ensure API key is valid.")
        tokens = state.get("total_tokens", 0)
        cost = state.get("total_cost", 0.0)

    return {"draft": draft, "logs": logs, "total_tokens": tokens, "total_cost": cost}

def editor_agent(state: ResearchState) -> Dict[str, Any]:
    """Reviews the draft and applies 'finishing touches'."""
    draft = state.get("draft", "")
    logs = state.get("logs", [])
    
    logs.append("Editor Agent: Reviewing and formatting draft...")
    
    if "API Error" in draft or "Error:" in draft:
        # Pass through the error
        return {"final_report": draft, "logs": logs, "approval_status": "pending_approval"}

    try:
        # Use gpt-4o-mini for speed
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        prompt = f"""
        You are a meticulous Senior Editor. Please review the following draft.
        Fix any grammatical errors, heavily format it using bold headers or bullet points to make it visually aesthetic, and add a concluding thought.
        
        DRAFT:
        {draft}
        """
        with get_openai_callback() as cb:
            response = llm.invoke([SystemMessage(content="You are a strict and highly aesthetic editor."), HumanMessage(content=prompt)])
            final_report = response.content
            logs.append("Editor Agent: Optimization complete. Flagging for Human Review.")
            tokens = state.get("total_tokens", 0) + cb.total_tokens
            cost = state.get("total_cost", 0.0) + cb.total_cost
    except Exception as e:
        final_report = draft
        logs.append(f"Editor Agent: Edit failed, using raw draft.")
        tokens = state.get("total_tokens", 0)
        cost = state.get("total_cost", 0.0)

    return {
        "final_report": final_report, 
        "logs": logs, 
        "approval_status": "pending_approval",
        "total_tokens": tokens, 
        "total_cost": cost
    }

def finalizer_node(state: ResearchState) -> Dict[str, Any]:
    """Runs after Human Approval."""
    logs = state.get("logs", [])
    decision = state.get("approval_status")
    
    if decision == "approved":
        logs.append("Finalizer: 🟢 Human approved the report. Workflow marked as COMPLETE.")
    elif decision == "rejected":
        # Fallback if the user rejects the output
        logs.append("Finalizer: 🔴 Human rejected the report. Output withdrawn.")
    else:
        logs.append("Finalizer: ⚠️ Workflow finalized without active approval.")
        
    return {"logs": logs}

# ==========================================
# 3. Workflow Routing
# ==========================================
def create_multi_agent_pipeline():
    workflow = StateGraph(ResearchState)
    
    # Add Nodes
    workflow.add_node("etl_input", etl_preprocessor)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("editor", editor_agent)
    workflow.add_node("publish", finalizer_node)
    
    # Strict Sequential Flow
    workflow.add_edge(START, "etl_input")
    workflow.add_edge("etl_input", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "editor")
    
    # Conditional edge isn't needed here if the flow is linear to human review, 
    # but we interrupt BEFORE publish to get approval.
    workflow.add_edge("editor", "publish")
    workflow.add_edge("publish", END)
    
    # Persistence using SqliteSaver for deployment-readiness
    conn = sqlite3.connect("research.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # Compile with HITL constraint
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["publish"]
    )
