import os
import json
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

class FAQIndexer:
    def __init__(self, persist_directory="./faq_chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()

    def index_faq(self, json_path):
        """
        Load FAQ JSON and index it into ChromaDB.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        documents = []
        for entry in data:
            content = f"Question: {entry['question']}\nAnswer: {entry['answer']}"
            doc = Document(
                page_content=content,
                metadata={"question": entry['question'], "answer": entry['answer']}
            )
            documents.append(doc)

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return vectorstore

    def get_answer(self, query):
        """
        Manual RAG: Retrieve context and generate answer.
        """
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        # 1. Similarity Search
        docs = vectorstore.similarity_search(query, k=2)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 2. Prompt Construction
        prompt = f"""
        You are a friendly customer support bot. 
        Use the following FAQ context to answer the user's question. 
        If the answer isn't in the context, politely say you don't know and suggest contact support@company.com.

        CONTEXT:
        {context}

        USER QUESTION: {query}
        
        ANSWER:"""
        
        # 3. LLM Call
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
        response = llm.invoke(prompt)
        
        return {
            "answer": response.content,
            "sources": docs
        }
