import streamlit as st
import os
import tempfile
from indexer import MultiDocIndexer
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Multi-Doc Intelligence", page_icon="🧠", layout="wide")

st.title("🧠 Multi-Document Intelligence System")
st.markdown("""
Upload multiple file types (**PDF, CSV, TXT**) and ask questions that synthesize 
information across all your documents.
""")

# Check for API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 OpenAI API Key not found! Set it in your terminal before running.")
    st.stop()

# Initialize Indexer
indexer = MultiDocIndexer()

# Sidebar - Sources Status
st.sidebar.title("Indexed Documents")
if os.path.exists("./multi_doc_chroma_db"):
    st.sidebar.success("✅ Unified Knowledge Base active.")
else:
    st.sidebar.info("Upload documents to build the index.")

# File Upload (Multi)
uploaded_files = st.file_uploader(
    "Upload Documents (PDF, CSV, TXT)", 
    type=["pdf", "csv", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Index All Documents"):
        with st.spinner("Processing documents into unified vector store..."):
            tmp_paths = []
            for uploaded_file in uploaded_files:
                ext = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_paths.append(tmp_file.name)
            
            try:
                indexer.add_documents(tmp_paths)
                st.success(f"Indexed {len(uploaded_files)} documents successfully!")
                
                # Cleanup temp files
                for p in tmp_paths:
                    os.unlink(p)
            except Exception as e:
                st.error(f"Error during indexing: {str(e)}")

# Chat Interface
st.divider()
query = st.chat_input("Ask a question across all your documents...")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing sources..."):
            try:
                result = indexer.ask_question(query)
                st.markdown(result["answer"])
                
                with st.expander("📝 View Relevant Source Snippets"):
                    for doc in result["sources"]:
                        st.write(f"**From `{doc.metadata.get('source_name')}`**:")
                        st.info(doc.page_content)
            except Exception as e:
                st.error(f"Error during retrieval: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.caption("Supports: PyPDF, CSVLoader, TextLoader")
