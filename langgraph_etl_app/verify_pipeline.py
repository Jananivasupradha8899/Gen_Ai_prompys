import pandas as pd
from pipeline import create_pipeline

def test_pipeline():
    # Load sample data
    df = pd.read_csv("d:/day3/langgraph_etl_app/sample_data.csv")
    
    # Configure parameters
    config = {
        "drop_na": False,
        "fill_na_strategy": "mean",
        "normalize": True,
        "age_threshold": 25
    }
    
    # Initialize state
    initial_state = {
        "raw_df": df,
        "processed_df": pd.DataFrame(),
        "config": config,
        "logs": [],
        "metrics": {}
    }
    
    # Run pipeline
    pipeline = create_pipeline()
    result = pipeline.invoke(initial_state)
    
    # Print results
    print("\n--- Pipeline Logs ---")
    for log in result["logs"]:
        print(log)
        
    print("\n--- Final Metrics ---")
    print(result["metrics"])
    
    print("\n--- Processed Data (First 5 Rows) ---")
    print(result["processed_df"].head())

if __name__ == "__main__":
    test_pipeline()
