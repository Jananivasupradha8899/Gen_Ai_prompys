from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. Define the State
class ModerationState(TypedDict):
    content: str
    is_flagged: bool
    review_decision: str  # "approved", "rejected", or "pending"
    logs: List[str]

# 2. Define Nodes
def check_content_node(state: ModerationState) -> Dict[str, Any]:
    content = state["content"].lower()
    logs = state.get("logs", [])
    logs.append("Phase 1: Automated content scanning...")
    
    # Simple Mock Moderation Logic
    flagged_words = ["offensive", "spam", "hate", "scam"]
    is_flagged = any(word in content for word in flagged_words)
    
    if is_flagged:
        logs.append("⚠️ Content flagged for manual review.")
    else:
        logs.append("✅ Content passed automated checks.")
        
    return {"is_flagged": is_flagged, "logs": logs, "review_decision": "pending"}

def human_review_node(state: ModerationState) -> Dict[str, Any]:
    # This node is reached only if is_flagged is true.
    # It serves as a placeholder for the Human decision.
    logs = state.get("logs", [])
    decision = state.get("review_decision", "pending")
    logs.append(f"Phase 2: Human review complete. Decision: {decision.upper()}")
    return {"logs": logs}

def finalize_node(state: ModerationState) -> Dict[str, Any]:
    logs = state.get("logs", [])
    decision = state["review_decision"]
    is_flagged = state["is_flagged"]
    
    if not is_flagged or decision == "approved":
        logs.append("🚀 Result: Content PUBLISHED successfully.")
    else:
        logs.append("🚫 Result: Content REJECTED and deleted.")
        
    return {"logs": logs}

# 3. Define the Router
def route_after_check(state: ModerationState):
    if state["is_flagged"]:
        return "human_review"
    return "finalize"

from langgraph.checkpoint.sqlite import SqliteSaver

# ... (States and Nodes remain the same) ...

# 4. Build and Compile the Graph
def create_moderation_graph(conn_string: str = "checkpoints.db"):
    workflow = StateGraph(ModerationState)
    
    workflow.add_node("check_content", check_content_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("finalize", finalize_node)
    
    workflow.add_edge(START, "check_content")
    
    workflow.add_conditional_edges(
        "check_content",
        route_after_check,
        {
            "human_review": "human_review",
            "finalize": "finalize"
        }
    )
    
    workflow.add_edge("human_review", "finalize")
    workflow.add_edge("finalize", END)
    
    # We return the workflow and use the checkpointer in the server
    return workflow
