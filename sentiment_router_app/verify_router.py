from router import create_router_workflow

def test_routing():
    workflow = create_router_workflow()
    
    test_queries = [
        "I am absolutely thrilled with this amazing product!",
        "I am very disappointed and angry about the delay.",
        "The package arrived at 3 PM today."
    ]
    
    for query in test_queries:
        print(f"\n--- Testing Query: '{query}' ---")
        initial_state = {
            "query": query,
            "sentiment": "",
            "confidence": 0.0,
            "response": "",
            "decision_path": []
        }
        
        results = workflow.invoke(initial_state)
        print(f"Sentiment: {results['sentiment']} ({results['confidence']:.2f})")
        print(f"Response: {results['response']}")
        print("Path:")
        for step in results['decision_path']:
            print(f"  - {step}")

if __name__ == "__main__":
    test_routing()
