import streamlit as st
from graph_logic import create_multi_agent_pipeline
import networkx as nx
import matplotlib.pyplot as plt
import os
import uuid
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ==========================================
# UI Configuration & Custom CSS
# ==========================================
st.set_page_config(page_title="AI Research Hub", layout="wide", page_icon="🧬")

st.markdown("""
<style>
    /* Light UI, Colorful Aesthetics */
    .stApp {
        background-color: #f7f9fa;
        color: #1a1a24;
    }
    h1, h2, h3 {
        color: #4a2deb; /* Vibrant purple */
    }
    .status-box {
        background-color: #ffffff;
        border-left: 5px solid #2db8eb; /* Cyan */
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .log-item {
        font-family: 'Courier New', monospace;
        background-color: #e8ecf1;
        padding: 5px;
        margin-bottom: 3px;
        border-radius: 4px;
        font-size: 0.85em;
    }
    .report-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 10px 15px rgba(0,0,0,0.05);
        border-top: 5px solid #4a2deb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if 'graph_instance' not in st.session_state:
    st.session_state.graph_instance = create_multi_agent_pipeline()
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'is_interrupted' not in st.session_state:
    st.session_state.is_interrupted = False
if 'latest_chat' not in st.session_state:
    st.session_state.latest_chat = None

graph = st.session_state.graph_instance
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ==========================================
# Sidebar: System Status & Graph Visuals
# ==========================================
with st.sidebar:
    st.title("🧬 System Telemetry")
    
    # API Key Warning
    if not os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") == "your_openai_api_key_here":
        st.error("⚠️ OPENAI_API_KEY is missing/invalid in .env file. The application will fail.")
        st.info("Please update the .env file in the app directory.")
    else:
        st.success("✅ OPENAI_API_KEY Connected")
        
    st.divider()

    st.subheader("⚙️ Deployment Controls")
    search_depth = st.slider("Deep Research Mode (Sources)", min_value=1, max_value=5, value=2, help="Higher values grab more context but run slower.")
    
    if st.button("🗑️ Clear Database History", use_container_width=True):
        try:
            if os.path.exists("research.db"):
                os.remove("research.db")
                st.toast("Database cleared successfully!")
                time.sleep(1)
                st.rerun()
        except:
            st.error("Could not delete DB. It might be in use.")

    st.divider()
    
    # Cost Telemetry
    st.subheader("💰 API Telemetry")
    t_tokens = 0
    t_cost = 0.0
    try:
        current_state = graph.get_state(config)
        if current_state and current_state.values:
            t_tokens = current_state.values.get("total_tokens", 0)
            t_cost = current_state.values.get("total_cost", 0.0)
    except:
        pass
    
    col_t, col_c = st.columns(2)
    col_t.metric("Tokens", f"{t_tokens:,}")
    col_c.metric("Cost", f"${t_cost:.4f}")
    
    st.divider()
    
    st.subheader("Architecture Map")
    st.caption("Directed Acyclic Graph (DAG) representing our agent execution path.")
    
    # Generate Workflow visualization using networkx and matplotlib
    try:
        fig, ax = plt.subplots(figsize=(4, 6))
        fig.patch.set_facecolor('#f7f9fa')
        
        G = nx.DiGraph()
        nodes = ["START", "ETL", "Researcher", "Writer", "Editor", "HITL Pause", "END"]
        edges = [("START", "ETL"), ("ETL", "Researcher"), ("Researcher", "Writer"), 
                 ("Writer", "Editor"), ("Editor", "HITL Pause"), ("HITL Pause", "END")]
        
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        pos = {
            "START": (0.5, 6), "ETL": (0.5, 5), "Researcher": (0.5, 4), 
            "Writer": (0.5, 3), "Editor": (0.5, 2), "HITL Pause": (0.5, 1), "END": (0.5, 0)
        }
        
        # Colors: nodes are distinct
        colors = ['#cccccc', '#2db8eb', '#eb2d69', '#eb8c2d', '#4a2deb', '#ffeb3b', '#cccccc']
        
        nx.draw(G, pos, ax=ax, with_labels=True, node_color=colors, 
                node_size=2000, font_size=9, font_weight="bold", font_color="white",
                edge_color="#4a2deb", arrowsize=15, arrowstyle="->")
        
        # Override specific text colors for readability
        nx.draw_networkx_labels(G, pos, labels={"START": "START", "END": "END"}, font_color="black")
        
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Map generation failed: {e}")

# ==========================================
# Main Dashboard
# ==========================================
st.title("Collaborative AI Insights Engine")
st.markdown("A multi-agent LangGraph workflow utilizing **DuckDuckGo Search**, **OpenAI Synthesis**, and **Human Checkpoints**.")

# 1. Inputs
query = st.text_input("Enter a complex technical topic to research:", placeholder="e.g., How do Transformers work in NLP?")

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("▶️ Launch Pipeline", use_container_width=True, type="primary"):
        if not query:
            st.warning("Please enter a query.")
        else:
            # Re-initialize thread to start fresh
            st.session_state.thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            initial_state = {
                "query": query, 
                "search_results": "", 
                "draft": "", 
                "editor_feedback": "", 
                "final_report": "",
                "approval_status": "in_progress",
                "search_depth": search_depth,
                "total_tokens": 0,
                "total_cost": 0.0,
                "logs": []
            }
            
            with st.spinner("🤖 Agents orchestrating..."):
                graph.invoke(initial_state, config=config)
                st.session_state.is_interrupted = True # Workflow hits interrupt_before=["publish"]
                st.rerun()

# 2. Tabs
tab_logs, tab_report, tab_hitl, tab_chat = st.tabs(["📋 Activity Logs", "📄 Iterative Reports", "🛑 Priority Override", "💬 App Assistant"])

# Fetch Current State
try:
    current_state = graph.get_state(config)
    values = current_state.values if current_state else {}
    next_nodes = current_state.next if current_state else []
except Exception:
    values = {}
    next_nodes = []

# --- Global Status Indicator ---
if values.get("query"):
    st.info(f"**Current Subject:** {values['query']}")

if values.get("approval_status"):
    status_map = {
        "in_progress": "🔄 Processing Agents...",
        "pending_approval": "🚨 Waiting for Human Approval",
        "approved": "✅ Officially Approved & Published",
        "rejected": "🚫 Rejected by Admin"
    }
    st.markdown(f"**Workflow Status:** `{status_map.get(values['approval_status'], values['approval_status'])}`")

with tab_logs:
    if values and "logs" in values:
        st.subheader("Process Trace")
        for log in values["logs"]:
            # Basic parsing to colorize prefixes
            components = log.split(":")
            if len(components) > 1:
                prefix = components[0]
                msg = ":".join(components[1:])
                st.markdown(f"<div class='log-item'><strong>{prefix}:</strong>{msg}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='log-item'>{log}</div>", unsafe_allow_html=True)
                
        if values.get("search_results"):
            with st.expander("🔍 View Raw Web Research Data"):
                st.text(values["search_results"])
    else:
        st.info("No activity logs available. Launch a pipeline.")

with tab_report:
    if values:
        col_draft, col_final = st.columns(2)
        with col_draft:
            st.subheader("📝 Writer's Initial Draft")
            draft_text = values.get("draft", "")
            if draft_text:
                st.markdown(f"<div class='status-box'>{draft_text}</div>", unsafe_allow_html=True)
            else:
                st.write("Draft pending...")
        
        with col_final:
            st.subheader("✨ Editor's Refined V2")
            final_text = values.get("final_report", "")
            if final_text:
                st.markdown(f"<div class='report-box'>{final_text}</div>", unsafe_allow_html=True)
            else:
                st.write("Refinement pending...")
    else:
        st.info("Reports will appear here once the Writer and Editor agents run.")

with tab_hitl:
    if "publish" in next_nodes:
        st.error("🚨 WORKFLOW PAUSED: Human Verification Required")
        st.markdown("The **Editor Agent** has finalized the document. Please review the *Iterative Reports* tab and decide if this meets quality standards.")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve for Publishing", use_container_width=True):
                # Update State
                graph.update_state(config, {"approval_status": "approved"})
                # Resume execution
                with st.spinner("Finalizing publish node..."):
                    graph.invoke(None, config=config)
                st.toast("Report Approved!")
                time.sleep(1)
                st.rerun()
                
        with c2:
            if st.button("🚫 Reject Content", use_container_width=True):
                 # Update State
                graph.update_state(config, {"approval_status": "rejected"})
                # Resume execution
                with st.spinner("Withdrawing publish request..."):
                    graph.invoke(None, config=config)
                st.error("Report Rejected.")
                time.sleep(1)
                st.rerun()
    elif values and "approval_status" in values:
        if values["approval_status"] == "approved":
            st.success("You have approved this report. Workflow Complete.")
            final_md = values.get("final_report", "")
            if final_md:
                st.download_button(
                    label="⬇️ Download Final Report (.md)",
                    data=final_md,
                    file_name=f"Research_Report_{uuid.uuid4().hex[:6]}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    type="primary"
                )
        elif values["approval_status"] == "rejected":
            st.warning("You rejected this report. Discarding output.")
        else:
            st.info("System is waiting for agents to finish their tasks.")
    else:
        st.info("Waiting for workflow to reach the Human Checkpoint.")

# --- Tab 4: App Assistant (Chatbot) ---
with tab_chat:
    st.subheader("💬 Ask About This App")
    st.markdown("I am a specialized assistant. Ask me to summarize the current Agent logs, explain the drafted text, or tell you what topic we are researching. **I have no memory of past questions.**")
    
    # Display the latest chat if it exists (but do not build a long history)
    if st.session_state.latest_chat:
        with st.chat_message("user"):
            st.markdown(st.session_state.latest_chat['user'])
        with st.chat_message("assistant"):
            st.markdown(st.session_state.latest_chat['assistant'])
            
    # Chat Input Box
    chat_prompt = st.chat_input("Ask a question strictly about the current pipeline...")
    
    if chat_prompt:
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("API Key missing.")
        else:
            with st.chat_message("user"):
                st.markdown(chat_prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing pipeline context..."):
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
                    
                    # Dump active state into prompt context safely
                    context_query = values.get("query", "None")
                    context_draft = values.get("draft", "None")
                    context_logs = "\n".join(values.get("logs", ["No logs yet."]))
                    
                    system_directive = f"""
                    You are a highly restrictive Assistant built exclusively into a Streamlit LangGraph app.
                    Your ONLY purpose is to answer questions about the user's ongoing research pipeline. 
                    
                    CRITICAL RULES:
                    1. If the user asks general-knowledge questions outside of the app's current context, politely but strictly REFUSE to answer.
                    2. If they ask for jokes, recipes, weather, code, or anything not related to the "Current Pipeline State", REFUSE.
                    3. Do not offer help outside analyzing the pipeline logs or drafts.
                    
                    === CURRENT PIPELINE STATE ===
                    Research Topic: {context_query}
                    Draft Summary: {context_draft}
                    Agent Internal Logs: {context_logs}
                    ==============================
                    """
                    
                    try:
                        response = llm.invoke([
                            SystemMessage(content=system_directive),
                            HumanMessage(content=chat_prompt)
                        ])
                        reply_text = response.content
                    except Exception as e:
                        reply_text = f"Failed to reach OpenAI: {e}"
                        
                    st.markdown(reply_text)
                    
                    # Save only the absolute latest message to state (no long history)
                    st.session_state.latest_chat = {
                        "user": chat_prompt,
                        "assistant": reply_text
                    }
