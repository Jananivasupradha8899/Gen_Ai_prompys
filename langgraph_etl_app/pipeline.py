from typing import TypedDict, Dict, Any, List
import pandas as pd
from langgraph.graph import StateGraph, START, END
from data_utils import clean_data, validate_data

# Define the State
class PipelineState(TypedDict):
    raw_df: pd.DataFrame
    processed_df: pd.DataFrame
    config: Dict[str, Any]
    logs: List[str]
    metrics: Dict[str, Any]

# 1. Extract Node
def extract_node(state: PipelineState) -> Dict[str, Any]:
    logs = state.get("logs", [])
    logs.append("Phase: Extracting data from source...")
    
    # In a real app, this might fetch from an API or DB
    # Here, we assume the raw_df is already provided via the initial state 
    # or we can reload it if a path was given.
    return {"logs": logs}

# 2. Transform Node
def transform_node(state: PipelineState) -> Dict[str, Any]:
    logs = state.get("logs", [])
    logs.append("Phase: Transforming and cleaning data...")
    
    processed_df = clean_data(state["raw_df"], state["config"])
    
    if processed_df.empty:
        logs.append("WARNING: Transformation resulted in an empty dataset. Check your filters.")
    else:
        logs.append(f"Transformation complete. Rows remaining: {len(processed_df)}")
    return {"processed_df": processed_df, "logs": logs}

# 3. Load Node
def load_node(state: PipelineState) -> Dict[str, Any]:
    logs = state.get("logs", [])
    logs.append("Phase: Loading data (final validation and metric gathering)...")
    
    metrics = validate_data(state["processed_df"])
    
    logs.append("Pipeline execution finished successfully.")
    return {"metrics": metrics, "logs": logs}

# Create the Graph
def create_pipeline() -> StateGraph:
    workflow = StateGraph(PipelineState)
    
    # Add Nodes
    workflow.add_node("extract", extract_node)
    workflow.add_node("transform", transform_node)
    workflow.add_node("load", load_node)
    
    # Add Edges (Linear Flow)
    workflow.add_edge(START, "extract")
    workflow.add_edge("extract", "transform")
    workflow.add_edge("transform", "load")
    workflow.add_edge("load", END)
    
    return workflow.compile()
