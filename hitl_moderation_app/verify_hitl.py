import uuid
from graph_logic import create_moderation_graph

def test_hitl_workflow():
    print("--- Starting HITL Isolation Test ---")
    graph = create_moderation_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # 1. Start with flagged content
    print("\n[Step 1] Submitting flagged content...")
    initial_state = {
        "content": "This is a spam message!",
        "is_flagged": False,
        "review_decision": "pending",
        "logs": []
    }
    
    # Invoke
    graph.invoke(initial_state, config=config)
    
    # Check current state
    state = graph.get_state(config)
    print(f"Current State Logs: {state.values['logs']}")
    print(f"Next Node: {state.next}")
    
    if "human_review" in state.next:
        print("✅ SUCCESS: Workflow correctly paused for human review.")
    else:
        print("❌ FAILURE: Workflow did not pause.")
        return

    # 2. Simulate Human Approval
    print("\n[Step 2] Simulating Human Approval...")
    graph.update_state(config, {"review_decision": "approved"})
    
    # Resume
    graph.invoke(None, config=config)
    
    # Check final state
    final_state = graph.get_state(config)
    print(f"Final State Logs: {final_state.values['logs']}")
    print(f"Next Node: {final_state.next}")
    
    if "PUBLISHED" in final_state.values['logs'][-1]:
        print("✅ SUCCESS: Workflow correctly resumed and published.")
    else:
        print("❌ FAILURE: Workflow did not finish correctly.")

if __name__ == "__main__":
    test_hitl_workflow()
