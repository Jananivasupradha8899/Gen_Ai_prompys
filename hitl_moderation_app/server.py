from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
from typing import List, Dict, Any
import json
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from graph_logic import create_moderation_graph

# Persistent Storage for graph checkpoints
# We use check_same_thread=False to allow multiple requests to access the DB
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph_builder = create_moderation_graph()

app = FastAPI(title="HITL Moderation API")
graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["human_review"])

# Persistent Storage for threads
THREADS_FILE = "threads.json"

def load_threads():
    if os.path.exists(THREADS_FILE):
        with open(THREADS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_threads(threads):
    with open(THREADS_FILE, "w") as f:
        json.dump(threads, f)

active_threads: Dict[str, Dict[str, Any]] = load_threads()

class ContentSubmission(BaseModel):
    content: str

class ReviewDecision(BaseModel):
    decision: str  # "approved" or "rejected"

@app.post("/submit")
async def submit_content(submission: ContentSubmission):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Start the workflow
    # This will run until it finish OR hit the 'human_review' interrupt
    initial_state = {
        "content": submission.content,
        "is_flagged": False,
        "review_decision": "pending",
        "logs": []
    }
    
    graph.invoke(initial_state, config=config)
    
    # Check current state to see if it's interrupted
    state = graph.get_state(config)
    is_pending = "human_review" in state.next
    
    active_threads[thread_id] = {
        "id": thread_id,
        "content": submission.content,
        "status": "pending" if is_pending else "completed"
    }
    save_threads(active_threads)
    
    return {
        "thread_id": thread_id,
        "status": "pending_approval" if is_pending else "processed",
        "message": "Content submitted for moderation."
    }

@app.get("/pending")
async def get_pending_reviews():
    """Returns a list of all threads waiting for human intervention."""
    pending = []
    for tid, data in active_threads.items():
        config = {"configurable": {"thread_id": tid}}
        state = graph.get_state(config)
        if "human_review" in state.next:
            pending.append(data)
    return pending

@app.get("/status/{thread_id}")
async def get_status(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state.values:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return {
        "values": state.values,
        "next_nodes": state.next,
        "metadata": active_threads.get(thread_id, {})
    }

@app.post("/action/{thread_id}")
async def take_action(thread_id: str, decision: ReviewDecision):
    if decision.decision not in ["approved", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid decision")
    
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    
    if "human_review" not in state.next:
        raise HTTPException(status_code=400, detail="Thread is not waiting for review")
    
    # Update state with the human decision
    # We use None as the input to resume, and pass the updated values
    graph.update_state(config, {"review_decision": decision.decision})
    
    # Resume execution
    graph.invoke(None, config=config)
    
    # Update our local tracker
    if thread_id in active_threads:
        active_threads[thread_id]["status"] = "completed"
    save_threads(active_threads)
        
    return {"message": f"Content {decision.decision} and workflow resumed."}

if __name__ == "__main__":
    import uvicorn
    import sys
    try:
        print("Starting HITL Moderation Server on http://127.0.0.1:8001")
        uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to start server: {e}")
        sys.exit(1)
