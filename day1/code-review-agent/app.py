import streamlit as st
import agent
import os
from dotenv import load_dotenv

# Page Config
st.set_page_config(page_title="Self-Reflecting Code Reviewer", page_icon="🤖", layout="wide")

# Sidebar - Configuration
st.sidebar.title("Settings")
mock_mode = st.sidebar.checkbox("Demo Mode (No API Key Required)", value=False)
user_api_key = st.sidebar.text_input("OpenAI API Key", type="password", disabled=mock_mode)

if mock_mode:
    api_key = "MOCK_MODE"
else:
    api_key = user_api_key or os.getenv("OPENAI_API_KEY")

st.title("🤖 Self-Reflecting Code Review Agent")
st.markdown("""
This app demonstrates an **Agentic Reflection Loop**. 
The agent doesn't just review your code once; it reviews its own feedback, critiques it, and improves it.
""")

# Input Section
code_input = st.text_area("Paste your Python code here:", height=200, placeholder="def example():\n    pass")

if st.button("Analyze Code"):
    if not api_key:
        st.error("Please provide an OpenAI API Key in the sidebar or .env file.")
    elif not code_input.strip():
        st.warning("Please enter some code to analyze.")
    else:
        # Step 1: AST Analysis
        with st.status("Performing Static Analysis (AST)...", expanded=True) as status:
            ast_results = agent.analyze_code_structure(code_input)
            
            if ast_results["errors"]:
                st.error("Syntax Errors Found:")
                for err in ast_results["errors"]:
                    st.write(f"- {err}")
                status.update(label="Analysis Completed with Errors", state="error")
            else:
                st.success("No syntax errors found!")
                if ast_results["smells"]:
                    st.info("Structural Smells Detected:")
                    for smell in ast_results["smells"]:
                        st.write(f"- {smell}")
                
                # Step 2: Reflection Loop
                st.write("---")
                st.subheader("Agent Reflection Loop")
                
                col1, col2, col3 = st.columns(3)
                
                # Placeholders for steps
                initial_placeholder = col1.empty()
                critique_placeholder = col2.empty()
                final_placeholder = col3.empty()
                
                status.update(label="LLM Reflection Loop in Progress...", state="running")
                
                for stage, content in agent.run_reflection_loop(code_input, api_key):
                    if stage == "initial":
                        initial_placeholder.markdown(f"### 1. Initial Review\n{content}")
                    elif stage == "critique":
                        critique_placeholder.markdown(f"### 2. Self-Critique\n{content}")
                    elif stage == "final":
                        final_placeholder.markdown(f"### 3. Improved Review\n{content}")
                
                status.update(label="All Steps Completed!", state="complete", expanded=False)

st.sidebar.markdown("---")
st.sidebar.info("""
### How it works:
1. **Parser**: Checks syntax using `ast`.
2. **Reviewer**: Generates initial feedback.
3. **Critic**: Evaluates the feedback for errors or gaps.
4. **Refiner**: Rewrites the review into a final version.
""")
