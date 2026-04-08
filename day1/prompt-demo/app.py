import streamlit as st
import os
from engine import ChatEngine
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Prompt Engineering Demo", page_icon="🧩", layout="wide")

# Sidebar for Configuration
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your key or set it in .env")
    st.markdown("---")
    st.info("""
    ### Patterns in this Demo:
    1. **CoT**: Think step-by-step.
    2. **ReAct**: Search mock database.
    3. **Reflection**: Critique and improve draft.
    """)

st.title("🧩 Advanced Prompt Engineering Chatbot")
st.write("Exposing the internal 'Reasoning' of the LLM using CoT, ReAct, and Self-Reflection.")

# Initialize Engine
engine = ChatEngine(api_key=api_key if api_key else None)

# Chat Layout
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main Column for Chat, Sidebar Column for Reasoning
col1, col2 = st.columns([2, 1])

with col1:
    chat_container = st.container(height=500)
    for msg in st.session_state.messages:
        with chat_container.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask about your order #123 or return policy..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with chat_container.chat_message("user"):
            st.markdown(query)

        with chat_container.chat_message("assistant"):
            with st.spinner("Processing (CoT → ReAct → Reflection)..."):
                result = engine.process_query(query)
                st.markdown(result["final_answer"])
                st.session_state.messages.append({"role": "assistant", "content": result["final_answer"]})
                st.session_state.last_reasoning = result["reasoning_steps"]

with col2:
    st.subheader("🧠 Model Reasoning")
    if "last_reasoning" in st.session_state:
        st.text_area("Internal Loop Log", value=st.session_state.last_reasoning, height=600)
    else:
        st.info("Ask a question to see the internal reasoning steps here.")

# Footer
st.markdown("---")
st.caption("Learning Demo: Built with Streamlit + OpenAI Mini")
