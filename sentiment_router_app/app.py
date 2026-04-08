import streamlit as st
import pandas as pd
import os
from router import create_router_workflow

# Set Page Config
st.set_page_config(page_title="Interactive Sentiment Router", layout="wide")

st.title("🧠 Interactive Sentiment Router")
st.markdown("""
This dashboard demonstrates **Dynamic LangGraph Configuration** and **LangSmith Observability**.
Adjust the sensitivity slider to see how the agent's routing logic changes in real-time.
""")

# --- Sidebar: Configuration & LangSmith ---
st.sidebar.header("⚙️ Graph Configuration")

# Sensitivity Slider
threshold = st.sidebar.slider(
    "Routing Sensitivity (Threshold)", 
    0.0, 1.0, 0.5, 0.05,
    help="Higher threshold means the agent requires more confidence to route to Positive/Negative handlers."
)

st.sidebar.divider()
st.sidebar.header("📊 LangSmith Observability")
# Set the provided key as the default for convenience
default_key = ""
ls_api_key = st.sidebar.text_input("LangSmith API Key", value=default_key, type="password")
ls_project = st.sidebar.text_input("Project Name", value="Sentiment-Router-Demo")

if st.sidebar.button("Enable LangSmith Tracing"):
    if ls_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = ls_api_key
        os.environ["LANGCHAIN_PROJECT"] = ls_project
        st.sidebar.success("Tracing Enabled! 🚀")
    else:
        st.sidebar.error("Please enter an API Key.")

# --- Main Application ---
user_input = st.text_input("Enter your message:", placeholder="e.g., I'm so happy with this service!")

if st.button("Route Query"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        with st.spinner("Executing LangGraph (with Dynamic Config)..."):
            # Initialize Workflow
            app = create_router_workflow()
            
            # Initial State
            initial_state = {
                "query": user_input,
                "sentiment": "",
                "confidence": 0.0,
                "response": "",
                "decision_path": []
            }
            
            # Execute with Dynamic Config
            config = {"configurable": {"threshold": threshold}, "recursion_limit": 10}
            results = app.invoke(initial_state, config=config)
            
            # --- Display Results ---
            st.divider()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🤖 Agent Response")
                st.chat_message("assistant").write(results["response"])
                
                st.subheader("🛤️ Decision Path")
                for step in results["decision_path"]:
                    st.code(step, language="markdown")
                
                if "Fallback" in "".join(results["decision_path"]):
                    st.warning(f"⚠️ Note: Routing was diverted to Fallback because confidence ({results['confidence']:.2f}) was lower than your sensitivity threshold ({threshold:.2f}).")
            
            with col2:
                st.subheader("📊 Sentiment Analysis")
                sentiment = results["sentiment"]
                
                st.metric("Detected Sentiment", sentiment.capitalize())
                st.progress(results["confidence"], text=f"Model Confidence: {results['confidence']:.2f}")
                st.caption(f"Configured Threshold: {threshold:.2f}")
                
                # Visual Indicator
                if sentiment == "positive":
                    st.success("Positive Vibe Detected! ✅")
                elif sentiment == "negative":
                    st.error("Negative Vibe Detected! ⚠️")
                else:
                    st.info("Neutral/Objective Vibe Detected. ℹ️")

st.divider()
st.caption("Architecture: LangGraph + Hugging Face Transformers. Tracing: LangSmith.")
