import streamlit as st
import os
from engine import HybridSearchEngine
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Hybrid Search Demo", page_icon="🔍", layout="wide")

st.title("🔍 Hybrid Search & Metadata Filtering")
st.markdown("""
Combine **Keyword Search** and **Vector Search** with real-time **Metadata Filters**.
Perfect for product catalogs and knowledge bases.
""")

# Check for API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 OpenAI API Key not found! Set it in your terminal as $env:OPENAI_API_KEY before running.")
    st.stop()

# Initialize Engine
engine = HybridSearchEngine()

# Sidebar for Filters
st.sidebar.title("Search Filters")
categories = ["All", "Electronics", "Furniture", "Outdoors"]
selected_category = st.sidebar.selectbox("Filter by Category", categories)

st.sidebar.markdown("---")
st.sidebar.info("""
### Match Types:
- **Keyword Match**: Exact match for your search term (Reliable/Direct).
- **Semantic Match**: Match based on meaning (Flexible/AI-powered).
""")

# Ingest Data if needed
if not os.path.exists("./hybrid_chroma_db"):
    with st.spinner("Indexing product catalog..."):
        engine.index_products("./data/products.json")
        st.success("✅ Catalog indexed!")

# Search Bar
query = st.text_input("Search for products (e.g., 'wireless', 'ergonomic', 'waterproof'):")

if query:
    with st.spinner(f"Searching for '{query}' in {selected_category}..."):
        results = engine.search(query, selected_category)
        
        if not results:
            st.warning("No products found matching your search and filter.")
        else:
            st.success(f"Found {len(results)} matches!")
            
            for item in results:
                p = item["doc"]
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.subheader(f"{p['name']} ({p['category']})")
                        st.write(item["content"])
                        st.markdown(f"**Match Type**: `{item['type']}`")
                    with col2:
                        st.title(f"${p['price']}")
                    st.divider()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with ChromaDB + OpenAI Embeddings")
