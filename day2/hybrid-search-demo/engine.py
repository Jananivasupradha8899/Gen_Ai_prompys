import os
import json
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

class HybridSearchEngine:
    def __init__(self, persist_directory="./hybrid_chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()

    def index_products(self, json_path):
        """
        Index products into ChromaDB with metadata.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        documents = []
        for p in data:
            # The text we'll embed is the name + description
            content = f"{p['name']}: {p['description']}"
            doc = Document(
                page_content=content,
                metadata={
                    "id": p["id"],
                    "name": p["name"],
                    "category": p["category"],
                    "price": p["price"]
                }
            )
            documents.append(doc)

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return vectorstore

    def search(self, query, category_filter=None):
        """
        Hybrid Search: Combined Metadata Filter + Vector Search + Keyword Search.
        """
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

        # 1. Prepare Metadata Filter
        where_filter = None
        if category_filter and category_filter != "All":
            where_filter = {"category": category_filter}

        # 2. Vector Search (Semantic)
        semantic_results = vectorstore.similarity_search_with_relevance_scores(
            query, 
            k=5, 
            filter=where_filter
        )

        # 3. Keyword Search (Manual check against names/descriptions)
        # In a larger app, this would use Chroma's native 'where_document' query
        keyword_results = []
        all_docs = vectorstore.get(where=where_filter, include=["documents", "metadatas"])
        
        for i, doc_text in enumerate(all_docs["documents"]):
            if query.lower() in doc_text.lower():
                # Score it as 1.0 (exact match)
                keyword_results.append({
                    "doc": all_docs["metadatas"][i],
                    "content": doc_text,
                    "type": "Keyword Match",
                    "score": 1.0
                })

        # 4. Process Semantic Results
        processed_semantic = []
        for doc, score in semantic_results:
            processed_semantic.append({
                "doc": doc.metadata,
                "content": doc.page_content,
                "type": "Semantic Match",
                "score": score
            })

        # 5. Merge and Rank
        # Combine lists and remove duplicates based on ID
        seen_ids = set()
        combined = []
        
        # Prioritize keyword matches, then semantic
        for item in keyword_results + processed_semantic:
            doc_id = item["doc"]["id"]
            if doc_id not in seen_ids:
                combined.append(item)
                seen_ids.add(doc_id)

        # Final sort by score
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined
