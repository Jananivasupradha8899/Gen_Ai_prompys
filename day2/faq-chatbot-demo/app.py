import streamlit as st
import os
from indexer import FAQIndexer
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="wide")

st.title("💬 Company FAQ Chatbot")
st.markdown("""
Ask questions about our company policies. This AI uses **RAG** to pull accurate answers 
from our official FAQ database.
""")

# Check for API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 OpenAI API Key not found! Set it in your terminal before running:")
    st.code('$env:OPENAI_API_KEY="your-key-here"', language="powershell")
    st.stop()

# Initialize Indexer
indexer = FAQIndexer()

# Auto-index FAQ if database doesn't exist
if not os.path.exists("./faq_chroma_db"):
    with st.spinner("Initializing FAQ Knowledge Base..."):
        indexer.index_faq("./data/faq.json")
        st.success("✅ Knowledge Base Ready!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = indexer.get_answer(prompt)
                full_response = result["answer"]
                st.markdown(full_response)
                
                # Show sources in an expander
                with st.expander("🔍 Related FAQ Entries"):
                    for doc in result["sources"]:
                        st.write(doc.page_content)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.info("""
### FAQ Data:
The chatbot is trained on:
- Return Policy
- Order Tracking
- International Shipping
- Customer Support Contact
- Payment Methods
""")
