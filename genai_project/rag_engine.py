import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class RAGEngine:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.documents = []  # Initialize documents list

    def load_documents(self, file_path: str) -> List[Document]:
        """Loads documents from a file path (PDF or TXT)."""
        print(f"DEBUG load_documents: Loading {file_path}")
        if file_path.endswith(".pdf"):
            # PyMuPDFLoader is more robust than PyPDFLoader
            loader = PyMuPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload .pdf or .txt")
        
        docs = loader.load()
        print(f"DEBUG load_documents: Loaded {len(docs)} pages, first page has {len(docs[0].page_content) if docs else 0} chars")
        return docs

    def process_documents(self, documents: List[Document]):
        """Splits documents and stores them in the vector database."""
        print(f"DEBUG process_documents: Received {len(documents)} documents")
        
        if not documents:
            print("DEBUG: No documents to process!")
            return
        
        # Check first document
        if len(documents) > 0:
            first_doc = documents[0]
            print(f"DEBUG: First document has {len(first_doc.page_content)} characters")
            print(f"DEBUG: First 100 chars: {first_doc.page_content[:100]}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        
        print(f"DEBUG: Starting split_documents...")
        splits = text_splitter.split_documents(documents)
        print(f"DEBUG: Number of splits: {len(splits)}")
        
        if len(splits) > 0:
            print(f"DEBUG: First split content: {splits[0].page_content[:100]}")
            try:
                test_embed = self.embeddings.embed_query("test")
                print(f"DEBUG: Test embedding success. Length: {len(test_embed)}")
            except Exception as e:
                print(f"DEBUG: Test embedding failed: {e}")

        if not splits:
            print("DEBUG: No splits created from documents.")
            return

        self.documents = splits # Store for test generation
        
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"DEBUG: Vector store created with {len(splits)} documents")
        # self.vector_store.persist() # Chroma 0.4+ persists automatically

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieves relevant documents for a given query."""
        if not self.vector_store:
            # Try to load existing DB
            if os.path.exists(self.persist_directory):
                 self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                return []
        
        return self.vector_store.similarity_search(query, k=k)

    def search_with_score(self, query: str, k: int = 4):
        """Retrieves documents with similarity scores."""
        if not self.vector_store:
            if os.path.exists(self.persist_directory):
                 self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                return []
        
        # Note: Chroma default metric is L2 distance (lower is better) or Cosine Distance
        return self.vector_store.similarity_search_with_score(query, k=k)

    def get_retriever(self):
        if not self.vector_store:
             if os.path.exists(self.persist_directory):
                 self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
        if self.vector_store:
            return self.vector_store.as_retriever()
        return None
    
    def load_multiple_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load multiple documents from a list of file paths.
        
        Args:
            file_paths: List of absolute paths to documents
            
        Returns:
            Combined list of all loaded documents
        """
        all_documents = []
        print(f"DEBUG load_multiple_documents: Processing {len(file_paths)} files")
        
        for file_path in file_paths:
            try:
                print(f"DEBUG: Loading file: {file_path}")
                docs = self.load_documents(file_path)
                print(f"DEBUG: Loaded {len(docs)} pages from {file_path}")
                
                # Add source metadata
                for doc in docs:
                    doc.metadata['source_file'] = os.path.basename(file_path)
                all_documents.extend(docs)
                print(f"DEBUG: Total documents so far: {len(all_documents)}")
                
            except Exception as e:
                print(f"ERROR loading {file_path}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"DEBUG: Returning {len(all_documents)} total documents")
        return all_documents
    
    def get_active_sources(self) -> List[str]:
        """
        Get list of unique source files in the current vector store.
        
        Returns:
            List of source filenames
        """
        if not self.documents:
            return []
        
        sources = set()
        for doc in self.documents:
            source = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
            sources.add(source)
        
        return sorted(list(sources))
