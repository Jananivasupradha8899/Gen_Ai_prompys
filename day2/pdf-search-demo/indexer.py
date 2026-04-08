import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

class PDFIndexer:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.embeddings = OpenAIEmbeddings()

    def process_pdf(self, pdf_path):
        """
        Extract text, split into chunks, and index into ChromaDB.
        """
        # 1. Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # 2. Split into chunks
        chunks = self.text_splitter.split_documents(documents)

        # 3. Create Vector Store (ChromaDB)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return vectorstore

    def ask_question(self, query):
        """
        Manually perform RAG to avoid dependency on problematic 'langchain.chains' module.
        """
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        # 1. Search for relevant documents
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 2. Construct Prompt
        prompt_text = f"Use the context below to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        # 3. Call LLM
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        response = llm.invoke(prompt_text)
        
        return {
            "result": response.content,
            "source_documents": docs
        }
