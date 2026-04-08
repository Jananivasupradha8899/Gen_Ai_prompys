import streamlit as st
import pandas as pd
import io
from pipeline import create_pipeline
from data_utils import validate_data

# Set Page Config
st.set_page_config(page_title="LangGraph ETL Pipeline", layout="wide")

st.title("🚀 LangGraph ETL Pipeline")
st.markdown("""
This application demonstrates a **linear data processing workflow** using **LangGraph** and **Pandas**. 
Upload a CSV, configure your transformation rules, and watch the data flow through the pipeline.
""")

# --- Sidebar: Configuration (Fine-Tuning) ---
st.sidebar.header("🛠 Pipeline Configuration")
drop_na = st.sidebar.checkbox("Drop rows with missing values", value=False)
fill_strategy = st.sidebar.selectbox("Fill NA Strategy (if not dropping)", ["mean", "median"])
normalize = st.sidebar.checkbox("Normalize numeric columns (0-1)", value=False)
age_threshold = st.sidebar.slider("Min Age Filter (if applicable)", 0, 100, 0)

config = {
    "drop_na": drop_na,
    "fill_na_strategy": fill_strategy,
    "normalize": normalize,
    "age_threshold": age_threshold
}

# --- Main: File Upload ---
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📥 Input Data Preview")
    st.dataframe(df.head(10))
    
    if st.button("Run ETL Pipeline"):
        # Initialize Pipeline
        pipeline = create_pipeline()
        
        # Initial State
        initial_state = {
            "raw_df": df,
            "processed_df": pd.DataFrame(),
            "config": config,
            "logs": [],
            "metrics": {}
        }
        
        with st.status("Executing Pipeline Nodes...", expanded=True) as status:
            # Run the workflow
            result = pipeline.invoke(initial_state)
            
            # Display Logs
            for log in result["logs"]:
                st.write(f"✅ {log}")
            
            status.update(label="Pipeline Finished!", state="complete", expanded=False)

        # --- Display Results ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✨ Transformed Data")
            st.dataframe(result["processed_df"])
        
        with col2:
            st.subheader("📊 Pipeline Metrics")
            st.json(result["metrics"])
            
        # Download Button
        csv_buffer = io.StringIO()
        result["processed_df"].to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Cleaned CSV",
            data=csv_buffer.getvalue(),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a CSV file to begin. You can use the 'sample_data.csv' in the project folder.")
    
    # Load sample button
    if st.button("Load Sample Data"):
        df_sample = pd.read_csv("d:/day3/langgraph_etl_app/sample_data.csv")
        st.subheader("📥 Sample Data Preview")
        st.dataframe(df_sample)
        
        # We can't easily re-trigger the full logic here without refreshing, 
        # but this shows the user what the data looks like.
        st.warning("Click 'Run ETL Pipeline' after loading (or upload the file manually).")
