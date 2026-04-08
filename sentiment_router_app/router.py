from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from sentiment_engine import analyze_query

# 1. Define the State
class RouterState(TypedDict):
    query: str
    sentiment: str
    confidence: float
    threshold: float
    response: str
    decision_path: List[str]

# 2. Define Nodes
def analyzer_node(state: RouterState, config: Any = None) -> Dict[str, Any]:
    query = state["query"]
    sentiment, score = analyze_query(query)
    
    # Read threshold from config
    # Some versions pass config as a dict, others as RunnableConfig
    if config and hasattr(config, "get"):
        threshold = config.get("configurable", {}).get("threshold", 0.5)
    elif config and hasattr(config, "configurable"):
        threshold = getattr(config, "configurable", {}).get("threshold", 0.5)
    else:
        threshold = 0.5
    
    path = state.get("decision_path", [])
    path.append(f"Analysis: Detected sentiment '{sentiment}' with confidence {score:.2f}")
    path.append(f"Config: Sensitivity set to {threshold:.2f}")
    
    return {"sentiment": sentiment, "confidence": score, "threshold": threshold, "decision_path": path}

def positive_handler(state: RouterState) -> Dict[str, Any]:
    path = state.get("decision_path", [])
    path.append("Branch: Positive Handler reached.")
    response = "That's wonderful to hear! 🌟 How can I further assist with your request?"
    return {"response": response, "decision_path": path}

def negative_handler(state: RouterState) -> Dict[str, Any]:
    path = state.get("decision_path", [])
    path.append("Branch: Negative Handler reached.")
    response = "I'm sorry you're feeling this way. 😔 I'm here to listen. Tell me more about what's bothering you."
    return {"response": response, "decision_path": path}

def neutral_handler(state: RouterState) -> Dict[str, Any]:
    path = state.get("decision_path", [])
    path.append("Branch: Neutral Handler reached.")
    response = "Understood. 📝 Please provide more details so I can help you objectively."
    return {"response": response, "decision_path": path}

def fallback_handler(state: RouterState) -> Dict[str, Any]:
    path = state.get("decision_path", [])
    path.append("Branch: Fallback (Low Confidence) reached.")
    response = "I'm picking up some emotion, but I'm not confident enough to route it. 🧐 Could you please clarify your tone or request?"
    return {"response": response, "decision_path": path}

# 3. Define Routing Logic
def route_sentiment(state: RouterState):
    """
    Decides which handler to call based on sentiment and the threshold saved in state.
    """
    sentiment = state["sentiment"]
    confidence = state["confidence"]
    threshold = state.get("threshold", 0.5)
    
    # If confidence is too low, go to fallback
    if confidence < threshold and sentiment != "neutral":
        return "fallback"
    
    if sentiment == "positive":
        return "positive"
    elif sentiment == "negative":
        return "negative"
    else:
        return "neutral"

# 4. Build the Graph
def create_router_workflow() -> StateGraph:
    workflow = StateGraph(RouterState)
    
    # Add Nodes
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("positive_handler", positive_handler)
    workflow.add_node("negative_handler", negative_handler)
    workflow.add_node("neutral_handler", neutral_handler)
    workflow.add_node("fallback_handler", fallback_handler)
    
    # Add Edges
    workflow.add_edge(START, "analyzer")
    
    # Conditional Branching
    workflow.add_conditional_edges(
        "analyzer",
        route_sentiment,
        {
            "positive": "positive_handler",
            "negative": "negative_handler",
            "neutral": "neutral_handler",
            "fallback": "fallback_handler"
        }
    )
    
    workflow.add_edge("positive_handler", END)
    workflow.add_edge("negative_handler", END)
    workflow.add_edge("neutral_handler", END)
    workflow.add_edge("fallback_handler", END)
    
    return workflow.compile()
