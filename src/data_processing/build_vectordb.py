"""
Build vector database for RAG (Retrieval-Augmented Generation)
"""

import sys
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config
from src.utils.helpers import load_json, chunk_text


class VectorDBBuilder:
    """Build and manage ChromaDB vector database"""
    
    def __init__(self, persist_dir: Path = None):
        if persist_dir is None:
            persist_dir = config.vectordb_dir / "course_materials"
        
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(config.embedding_model)
        print(f"✓ Loaded {config.embedding_model}")
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="course_materials",
            metadata={"description": "Operating Systems and Networks course materials"}
        )
    
    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to vector database"""
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ Added {len(documents)} documents to vector database")
    
    def process_slides(self):
        """Process slides and add to vector DB"""
        slides_file = config.data_dir / "processed" / "slides" / "all_slides_combined.json"
        
        if not slides_file.exists():
            print("⚠️  No slides data found")
            return 0
        
        data = load_json(slides_file)
        documents = []
        metadatas = []
        ids = []
        
        for file_data in data['files']:
            filename = file_data['filename']
            for slide in file_data['slides']:
                content = slide['content']
                if len(content) > 50:
                    doc_id = f"slide_{filename}_{slide['slide_number']}"
                    documents.append(content)
                    metadatas.append({
                        "source": "slide",
                        "filename": filename,
                        "slide_number": slide['slide_number'],
                        "type": "lecture_slide"
                    })
                    ids.append(doc_id)
        
        if documents:
            self.add_documents(documents, metadatas, ids)
            print(f"  📊 Processed {len(documents)} slides")
        
        return len(documents)
    
    def process_books(self):
        """Process books and add to vector DB"""
        books_file = config.data_dir / "processed" / "books" / "all_pdfs_combined.json"
        
        if not books_file.exists():
            print("⚠️  No PDF data found")
            return 0
        
        data = load_json(books_file)
        documents = []
        metadatas = []
        ids = []
        
        for file_data in data['files']:
            filename = file_data['filename']
            for i, chunk in enumerate(file_data['chunks'], 1):
                if len(chunk) > 100:
                    doc_id = f"book_{filename}_{i}"
                    documents.append(chunk)
                    metadatas.append({
                        "source": "book",
                        "filename": filename,
                        "chunk_number": i,
                        "type": "textbook"
                    })
                    ids.append(doc_id)
        
        if documents:
            self.add_documents(documents, metadatas, ids)
            print(f"  📚 Processed {len(documents)} book chunks")
        
        return len(documents)
    
    def get_stats(self):
        """Get vector database statistics"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection.name
        }


def main():
    """Main execution function"""
    print("=" * 60)
    print("Building Vector Database for RAG".center(60))
    print("=" * 60)
    
    # Initialize builder
    builder = VectorDBBuilder()
    
    # Check if collection already has data
    existing_count = builder.collection.count()
    if existing_count > 0:
        print(f"\n⚠️  Vector database already contains {existing_count} documents")
        response = input("Do you want to rebuild? (y/n): ").lower()
        if response != 'y':
            print("Cancelled.")
            return
        
        # Delete and recreate collection
        builder.client.delete_collection("course_materials")
        builder.collection = builder.client.create_collection(
            name="course_materials",
            metadata={"description": "Operating Systems and Networks course materials"}
        )
    
    # Process materials
    total_docs = 0
    total_docs += builder.process_slides()
    total_docs += builder.process_books()
    
    # Show statistics
    stats = builder.get_stats()
    print(f"\n✅ Vector database built successfully!")
    print(f"📊 Total documents: {stats['total_documents']}")
    print(f"📁 Location: {config.vectordb_dir / 'course_materials'}")
    
    if total_docs == 0:
        print("\n⚠️  No documents were added!")
        print("Please ensure you have run:")
        print("  1. extract_slides.py")
        print("  2. extract_pdfs.py")


if __name__ == "__main__":
    main()
