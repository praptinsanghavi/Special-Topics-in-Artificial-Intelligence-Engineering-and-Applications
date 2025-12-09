"""
Document Library Manager

Manages multiple uploaded documents and allows users to select which documents
the AI should use as its knowledge base. Provides explicit control over RAG sources.
"""

import os
import json
from typing import List, Dict
from datetime import datetime


class DocumentLibrary:
    """
    Manages a library of uploaded documents.
    
    Features:
        - Store multiple documents with metadata
        - Select/deselect documents for active use
        - Track document upload dates and sizes
        - Persist library state
    """
    
    def __init__(self, library_file: str = "document_library.json"):
        self.library_file = library_file
        self.documents = self._load_library()
    
    def add_document(self, filename: str, filepath: str, num_chunks: int) -> Dict:
        """
        Add a document to the library.
        
        Args:
            filename: Original filename
            filepath: Path to stored file
            num_chunks: Number of chunks after processing
            
        Returns:
            Document metadata dict
        """
        doc_id = f"doc_{len(self.documents) + 1}"
        
        doc_metadata = {
            "id": doc_id,
            "filename": filename,
            "filepath": filepath,
            "num_chunks": num_chunks,
            "upload_date": datetime.now().isoformat(),
            "active": True,  # New documents are active by default
            "file_size": os.path.getsize(filepath) if os.path.exists(filepath) else 0
        }
        
        self.documents[doc_id] = doc_metadata
        self._save_library()
        return doc_metadata
    
    def get_active_documents(self) -> List[Dict]:
        """Get list of currently active documents."""
        return [doc for doc in self.documents.values() if doc.get("active", False)]
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents in library."""
        return list(self.documents.values())
    
    def toggle_document(self, doc_id: str) -> bool:
        """Toggle document active status."""
        if doc_id in self.documents:
            self.documents[doc_id]["active"] = not self.documents[doc_id].get("active", False)
            self._save_library()
            return self.documents[doc_id]["active"]
        return False
    
    def remove_document(self, doc_id: str):
        """Remove document from library."""
        if doc_id in self.documents:
            # Delete file if it exists
            filepath = self.documents[doc_id].get("filepath")
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
            
            del self.documents[doc_id]
            self._save_library()
    
    def get_active_filepaths(self) -> List[str]:
        """Get filepaths of all active documents."""
        active_docs = self.get_active_documents()
        print(f"DEBUG get_active_filepaths: Found {len(active_docs)} active documents")
        
        result = []
        for doc in active_docs:
            filepath = doc["filepath"]
            exists = os.path.exists(filepath)
            print(f"DEBUG: File '{filepath}' exists={exists}")
            if exists:
                result.append(filepath)
        
        print(f"DEBUG: Returning {len(result)} filepaths: {result}")
        return result
    
    def _load_library(self) -> Dict:
        """Load library from JSON file."""
        if os.path.exists(self.library_file):
            try:
                with open(self.library_file, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_library(self):
        """Save library to JSON file."""
        with open(self.library_file, "w") as f:
            json.dump(self.documents, f, indent=2)
