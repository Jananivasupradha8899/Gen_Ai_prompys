import os
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

class MultiDocIndexer:
    def __init__(self, persist_directory="./multi_doc_chroma_db"):
        self.persist_directory = persist_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        self.embeddings = OpenAIEmbeddings()

    def load_document(self, file_path):
        """
        Dynamically load documents based on file extension.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.csv':
            loader = CSVLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return loader.load()

    def add_documents(self, file_paths):
        """
        Process and index a list of files.
        """
        all_docs = []
        for path in file_paths:
            docs = self.load_document(path)
            # Add metadata about source
            for doc in docs:
                doc.metadata["source_name"] = os.path.basename(path)
            all_docs.extend(docs)

        # Split into chunks
        chunks = self.text_splitter.split_documents(all_docs)

        # Update Vector Store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return vectorstore

    def ask_question(self, query):
        """
        Manual RAG across all documents.
        """
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        # 1. Retrieve top 5 relevance documents
        docs = vectorstore.similarity_search(query, k=5)
        
        # 2. Build Context
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source_name", "Unknown")
            context_parts.append(f"--- Source {i+1} ({source}) ---\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # 3. Call LLM
        prompt = f"""
        You are an intelligent document analyst. 
        Answer the following question using ONLY the provided context from multiple documents.
        If the answer isn't in the context, say you don't know. 
        List which source files you are using for the answer.

        CONTEXT:
        {context}

        QUESTION: {query}
        
        ANSWER:"""
        
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        response = llm.invoke(prompt)
        
        return {
            "answer": response.content,
            "sources": docs
        }
