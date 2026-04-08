import sqlite3
import json
from langgraph.checkpoint.sqlite import SqliteSaver

def inspect_checkpoints():
    print("--- 🧐 HITL Database Inspector ---")
    conn = sqlite3.connect("checkpoints.db")
    cur = conn.cursor()
    
    # Check if there are any records
    cur.execute("SELECT count(*) FROM checkpoints")
    count = cur.fetchone()[0]
    
    if count == 0:
        print("Empty Database: No checkpoints saved yet. Try submitting content in the dashboard first!")
        return

    print(f"Total Checkpoints Found: {count}")
    
    # List unique threads
    cur.execute("SELECT DISTINCT thread_id FROM checkpoints")
    threads = cur.fetchall()
    
    print("\nActive Moderation Threads:")
    for i, (tid,) in enumerate(threads):
        print(f" {i+1}. Thread ID: {tid}")

    # Peek into the latest state
    # We use SqliteSaver to properly 'decode' the binary blobs
    saver = SqliteSaver(conn)
    for (tid,) in threads:
        config = {"configurable": {"thread_id": tid}}
        state_data = saver.get(config)
        
        if state_data:
            print(f"\n--- Details for Thread {tid[:8]}... ---")
            
            # Handle either CheckpointTuple (object) or Dict
            if hasattr(state_data, "checkpoint"):
                checkpoint = state_data.checkpoint
            elif isinstance(state_data, dict) and "checkpoint" in state_data:
                checkpoint = state_data["checkpoint"]
            else:
                checkpoint = state_data
            
            # Extract values
            if isinstance(checkpoint, dict):
                values = checkpoint.get("values", {})
            elif hasattr(checkpoint, "values"):
                values = checkpoint.values
            else:
                values = {}
                
            print(f"Content: \"{values.get('content', 'N/A')}\"")
            print(f"Decision: {str(values.get('review_decision', 'N/A')).upper()}")
            print(f"Flagged?: {values.get('is_flagged', 'N/A')}")
            print("Logs:")
            for log in values.get('logs', []):
                print(f"  - {log}")
        else:
            print(f"[-] Thread {tid[:8]}: No state found.")

    conn.close()

if __name__ == "__main__":
    inspect_checkpoints()
