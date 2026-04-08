import streamlit as st
import os
import tempfile
from indexer import PDFIndexer
from dotenv import load_dotenv

# Load .env only if it exists (for local devs), otherwise relies on environment variables
load_dotenv()

st.set_page_config(page_title="PDF AI Search", page_icon="📄", layout="wide")

st.title("📄 PDF Intelligent Search & Indexing")
st.markdown("""
Upload a PDF and ask questions about its content. This app uses **ChromaDB** for vector storage 
and **LangChain** for the RAG pipeline.
""")

# Check for API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("⚠️ OpenAI API Key not found in environment! Please set it in your terminal or .env file.")
    st.code("set OPENAI_API_KEY=sk-xxxx (Windows)\nexport OPENAI_API_KEY=sk-xxxx (Linux/Mac)")

# Initialize Indexer
indexer = PDFIndexer(persist_directory="./chroma_db")

# Sidebar - Progress & Stats
st.sidebar.title("Index Stats")
if os.path.exists("./chroma_db"):
    st.sidebar.success("✅ Multi-document index detected.")
else:
    st.sidebar.info("No index found. Upload a PDF to begin.")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    if st.button("Index Document"):
        with st.spinner("Processing PDF... (Extraction -> Chunking -> Embedding)"):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                indexer.process_pdf(tmp_path)
                st.success(f"Successfully indexed '{uploaded_file.name}'!")
                os.unlink(tmp_path) # Clean up temp file
            except Exception as e:
                st.error(f"Error during indexing: {str(e)}")

# Query Section
st.divider()
query = st.text_input("Ask a question about your indexed documents:")

if query:
    if not api_key:
        st.error("Cannot perform search without an API Key.")
    else:
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            try:
                response = indexer.ask_question(query)
                
                st.subheader("🤖 AI Assistant:")
                st.write(response["result"])
                
                with st.expander("📚 View Source References"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'N/A') + 1}):**")
                        st.info(doc.page_content)
            except Exception as e:
                st.error(f"Error during retrieval: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.caption("Powered by ChromaDB + LangChain")
