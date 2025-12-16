"""
RAG System using ChromaDB and fine-tuned model
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config


class RAGSystem:
    """Retrieval-Augmented Generation System"""
    
    def __init__(self, persist_dir: Path = None):
        if persist_dir is None:
            persist_dir = config.vectordb_dir / "course_materials"
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collection
        try:
            self.collection = self.client.get_collection("course_materials")
            print(f"✓ Loaded vector database ({self.collection.count()} documents)")
        except:
            raise ValueError(f"Vector database not found at {persist_dir}. Run build_vectordb.py first.")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(config.embedding_model)
    
    def retrieve(self, query: str, n_results: int = 5, filter_dict: Dict = None) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filter_dict
        )
        
        # Format results
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            retrieved_docs.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return retrieved_docs
    
    def get_context(self, query: str, n_results: int = 3, max_length: int = 1500) -> Tuple[str, List[Dict]]:
        """Get context for RAG"""
        
        # Retrieve relevant documents
        docs = self.retrieve(query, n_results=n_results)
        
        # Combine documents into context
        context_parts = []
        sources = []
        current_length = 0
        
        for doc in docs:
            text = doc['document']
            metadata = doc['metadata']
            
            # Add document with source
            source_info = f"[{metadata.get('source', 'unknown')}] {metadata.get('filename', '')}"
            if 'slide_number' in metadata:
                source_info += f" - Slide {metadata['slide_number']}"
            elif 'chunk_number' in metadata:
                source_info += f" - Section {metadata['chunk_number']}"
            
            # Check length
            if current_length + len(text) > max_length:
                # Truncate if needed
                remaining = max_length - current_length
                if remaining > 100:  # Only add if significant space left
                    text = text[:remaining] + "..."
                    context_parts.append(f"{source_info}:\n{text}")
                    sources.append(source_info)
                break
            
            context_parts.append(f"{source_info}:\n{text}")
            sources.append(source_info)
            current_length += len(text)
        
        context = "\n\n".join(context_parts)
        
        return context, sources
    
    def search_by_topic(self, topic: str, n_results: int = 10) -> List[Dict]:
        """Search documents by topic/keyword"""
        return self.retrieve(topic, n_results=n_results)
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.collection.name
        }


def main():
    """Test RAG system"""
    print("=" * 60)
    print("Testing RAG System".center(60))
    print("=" * 60)
    
    # Initialize RAG
    rag = RAGSystem()
    
    # Test query
    test_query = "What is process scheduling?"
    print(f"\nQuery: {test_query}")
    
    # Retrieve context
    context, sources = rag.get_context(test_query, n_results=3)
    
    print(f"\n📚 Retrieved {len(sources)} sources:")
    for i, source in enumerate(sources, 1):
        print(f"  {i}. {source}")
    
    print(f"\n📄 Context (first 500 chars):")
    print(context[:500] + "..." if len(context) > 500 else context)
    
    # Show stats
    stats = rag.get_stats()
    print(f"\n📊 Database stats:")
    print(f"  Total documents: {stats['total_documents']}")


if __name__ == "__main__":
    main()
