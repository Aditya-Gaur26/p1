"""
Enhanced RAG System with hybrid retrieval, reranking, and query expansion
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config
from src.inference.hybrid_retriever import HybridRetriever, QueryExpander


class RAGSystem:
    """Enhanced Retrieval-Augmented Generation System with hybrid retrieval"""
    
    def __init__(self, persist_dir: Path = None, use_hybrid: bool = True, 
                 use_rerank: bool = True, use_query_expansion: bool = True):
        """
        Initialize RAG System
        
        Args:
            persist_dir: Path to vector database
            use_hybrid: Use hybrid retrieval (dense + sparse)
            use_rerank: Use reranking
            use_query_expansion: Expand queries with synonyms
        """
        if persist_dir is None:
            persist_dir = config.vectordb_dir / "course_materials"
        
        self.use_hybrid = use_hybrid
        self.use_rerank = use_rerank
        self.use_query_expansion = use_query_expansion
        
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
        
        # Initialize embedding model (upgraded to mpnet)
        embedding_model_name = getattr(config, 'embedding_model_upgraded', config.embedding_model)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize query expander
        if use_query_expansion:
            self.query_expander = QueryExpander()
        
        # Initialize hybrid retriever if needed
        if use_hybrid:
            self._init_hybrid_retriever()
    
    def _init_hybrid_retriever(self):
        """Initialize hybrid retriever with all documents"""
        print("🔄 Initializing hybrid retriever...")
        
        # Get all documents from ChromaDB
        all_data = self.collection.get(include=['documents', 'metadatas'])
        
        # Prepare documents for hybrid retriever
        documents = []
        for i, (doc, metadata) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
            documents.append({
                'id': all_data['ids'][i] if 'ids' in all_data else str(i),
                'content': doc,
                'metadata': metadata
            })
        
        # Create hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            documents=documents,
            embedding_model=self.embedding_model,
            use_rerank=self.use_rerank
        )
        
        print(f"✓ Hybrid retriever initialized with {len(documents)} documents")
    
    def retrieve(self, query: str, n_results: int = 5, filter_dict: Dict = None, 
                 method: str = "auto") -> List[Dict]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query string
            n_results: Number of results to return
            filter_dict: Metadata filters
            method: "auto", "hybrid", "dense", or "sparse"
        
        Returns:
            List of retrieved documents
        """
        # Expand query if enabled
        original_query = query
        if self.use_query_expansion:
            query = self.query_expander.expand_query(query)
            if query != original_query:
                print(f"🔍 Expanded query: {query}")
        
        # Use hybrid retrieval if available and requested
        if method == "auto":
            method = "hybrid" if self.use_hybrid else "dense"
        
        if method != "dense" and self.use_hybrid:
            # Use hybrid retriever
            retrieved_docs = self.hybrid_retriever.retrieve(
                query, 
                top_k=n_results, 
                method=method
            )
        else:
            # Fallback to dense retrieval only
            query_embedding = self.embedding_model.encode([query])[0]
            
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
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': 1.0 - results['distances'][0][i] if 'distances' in results else 0.0,
                    'rank': i + 1
                })
        
        return retrieved_docs
    
    def estimate_query_complexity(self, query: str) -> float:
        """
        Estimate query complexity based on various factors
        
        Returns:
            Complexity score between 0 and 1
        """
        complexity = 0.0
        
        # Length-based complexity
        word_count = len(query.split())
        if word_count > 15:
            complexity += 0.3
        elif word_count > 8:
            complexity += 0.15
        
        # Check for complex question words
        complex_words = ['why', 'how', 'compare', 'analyze', 'evaluate', 'explain in detail']
        for word in complex_words:
            if word in query.lower():
                complexity += 0.2
                break
        
        # Check for technical terms
        if self.use_query_expansion:
            key_terms = self.query_expander.extract_key_terms(query)
            complexity += min(len(key_terms) * 0.15, 0.3)
        
        # Check for multiple questions
        if '?' in query and query.count('?') > 1:
            complexity += 0.2
        
        return min(complexity, 1.0)
    
    def get_context(self, query: str, n_results: int = None, max_length: int = None,
                   adaptive: bool = True) -> Tuple[str, List[Dict]]:
        """
        Get context for RAG with adaptive sizing
        
        Args:
            query: Query string
            n_results: Number of documents to retrieve (auto if None)
            max_length: Maximum context length (auto if None)
            adaptive: Adjust context based on query complexity
        
        Returns:
            Tuple of (context_string, sources_list)
        """
        # Adaptive context sizing
        if adaptive:
            complexity = self.estimate_query_complexity(query)
            if n_results is None:
                n_results = 5 if complexity > 0.6 else 3
            if max_length is None:
                max_length = int(2000 if complexity > 0.6 else 1500)
        else:
            n_results = n_results or 3
            max_length = max_length or 1500
        
        # Retrieve relevant documents
        docs = self.retrieve(query, n_results=n_results)
        
        # Combine documents into context
        context_parts = []
        sources = []
        current_length = 0
        
        for doc in docs:
            text = doc.get('content', doc.get('document', ''))
            metadata = doc.get('metadata', {})
            score = doc.get('score', 0.0)
            
            # Add document with source and relevance score
            source_info = f"[{metadata.get('source', 'unknown')}] {metadata.get('filename', '')}"
            if 'slide_number' in metadata:
                source_info += f" - Slide {metadata['slide_number']}"
            elif 'chunk_number' in metadata:
                source_info += f" - Section {metadata['chunk_number']}"
            
            # Add relevance indicator
            if score > 0.8:
                relevance = "★★★ Highly Relevant"
            elif score > 0.6:
                relevance = "★★ Relevant"
            else:
                relevance = "★ Related"
            
            # Check length
            if current_length + len(text) > max_length:
                # Truncate if needed
                remaining = max_length - current_length
                if remaining > 100:  # Only add if significant space left
                    text = text[:remaining] + "..."
                    context_parts.append(f"{source_info} ({relevance}):\n{text}")
                    sources.append(source_info)
                break
            
            context_parts.append(f"{source_info} ({relevance}):\n{text}")
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
