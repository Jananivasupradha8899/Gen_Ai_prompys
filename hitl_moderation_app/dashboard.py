import streamlit as st
import requests
import time
from datetime import datetime

# Config
API_URL = "http://localhost:8001"

st.set_page_config(page_title="Premium HITL Moderation", layout="wide", page_icon="🛡️")

# --- Custom Styles ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #121212 100%);
        color: #e0e0e0;
    }
    .status-badge {
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.8rem;
    }
    .pending { background-color: #ff9800; color: white; }
    .approved { background-color: #4caf50; color: white; }
    .rejected { background-color: #f44336; color: white; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Live Status & Auto-Poll Config ---
st.sidebar.title("🛡️ Admin Control")
st.sidebar.divider()

# Auto-poll toggle
auto_poll = st.sidebar.toggle("Auto-Poll (5s Heartbeat)", value=True)

def fetch_pending():
    try:
        res = requests.get(f"{API_URL}/pending", timeout=2)
        return res.json() if res.status_code == 200 else []
    except:
        return []

pending_items = fetch_pending()
pending_count = len(pending_items)

# Live Status Badge
if pending_count > 0:
    st.sidebar.error(f"🚨 {pending_count} Items Awaiting Review")
else:
    st.sidebar.success("✅ All Clear: No Pending Items")

st.sidebar.divider()
st.sidebar.caption(f"Last sync: {datetime.now().strftime('%H:%M:%S')}")

# --- Main Dashboard ---
st.title("🛡️ Premium HITL Moderation")

tab1, tab2 = st.tabs(["✍️ Content Submission", "👮 Moderator Dashboard"])

# --- Tab 1: Submit Content ---
with tab1:
    st.subheader("Submit Content")
    content_input = st.text_area("Content to analyze:", height=100, placeholder="Try words like 'spam', 'offensive', or 'scam'...")
    
    if st.button("🚀 Process through LangGraph", use_container_width=True):
        if content_input:
            with st.status("Analyzing content in LangGraph...") as status:
                try:
                    res = requests.post(f"{API_URL}/submit", json={"content": content_input})
                    if res.status_code == 200:
                        data = res.json()
                        if data["status"] == "pending_approval":
                            status.update(label="⚠️ Flagged for Human Review!", state="error")
                            st.warning(f"Thread `{data['thread_id']}` is now blocked until an admin approves it.")
                        else:
                            status.update(label="✅ Content Published Safely!", state="complete")
                            st.success("Automated check passed. Content is live.")
                    else:
                        st.error("Server error. Please ensure FastAPI is running on port 8001.")
                except Exception as e:
                    st.error(f"Connection error: {e}")
        else:
            st.info("Input some text to begin.")

# --- Tab 2: Moderator Dashboard ---
with tab2:
    st.subheader("Moderation Queue")
    
    if not pending_items:
        st.info("System is quiet. No flagged content detected.")
        st.balloons()
    else:
        for item in pending_items:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Thread ID:** `{item['id']}`")
                    st.markdown(f"> {item['content']}")
                
                with col2:
                    st.write("**Required Action:**")
                    if st.button("✅ Approve", key=f"app_{item['id']}", use_container_width=True):
                        requests.post(f"{API_URL}/action/{item['id']}", json={"decision": "approved"})
                        st.rerun()
                    
                    if st.button("🚫 Reject", key=f"rej_{item['id']}", use_container_width=True):
                        requests.post(f"{API_URL}/action/{item['id']}", json={"decision": "rejected"})
                        st.rerun()
                
                if st.button("📊 View Audit Logs", key=f"logs_{item['id']}"):
                    log_res = requests.get(f"{API_URL}/status/{item['id']}")
                    if log_res.status_code == 200:
                        st.json(log_res.json()["values"]["logs"])

# --- Heartbeat Logic (Auto-Poll) ---
if auto_poll:
    time.sleep(5)
    st.rerun()
