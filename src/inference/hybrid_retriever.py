"""
Enhanced hybrid retrieval system with reranking and query expansion
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


class HybridRetriever:
    """Combines dense and sparse retrieval with reranking"""
    
    def __init__(self, documents: List[Dict[str, Any]], 
                 embedding_model=None,
                 use_rerank: bool = True):
        """
        Initialize hybrid retriever
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            embedding_model: Sentence transformer model for dense retrieval
            use_rerank: Whether to use cross-encoder for reranking
        """
        self.documents = documents
        self.embedding_model = embedding_model
        self.use_rerank = use_rerank
        
        # Initialize BM25 for sparse retrieval
        self.corpus = [doc.get('content', '') for doc in documents]
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Initialize reranker if needed
        if use_rerank:
            try:
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except:
                print("⚠️  Could not load reranker, continuing without reranking")
                self.use_rerank = False
        
        # Precompute embeddings for dense retrieval
        if embedding_model is not None:
            print("📊 Computing document embeddings...")
            self.doc_embeddings = embedding_model.encode(
                self.corpus,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        else:
            self.doc_embeddings = None
    
    def retrieve_sparse(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """BM25 sparse retrieval"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = [(idx, scores[idx]) for idx in top_indices]
        
        return results
    
    def retrieve_dense(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Dense retrieval using embeddings"""
        if self.embedding_model is None or self.doc_embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        # Compute cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity([query_embedding], self.doc_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = [(idx, scores[idx]) for idx in top_indices]
        
        return results
    
    def reciprocal_rank_fusion(self, 
                               results_list: List[List[Tuple[int, float]]],
                               k: int = 60) -> List[Tuple[int, float]]:
        """
        Combine multiple ranking lists using Reciprocal Rank Fusion (RRF)
        
        Args:
            results_list: List of result lists, each containing (doc_id, score) tuples
            k: Constant for RRF formula (typically 60)
        
        Returns:
            Combined ranked list
        """
        rrf_scores = {}
        
        for results in results_list:
            for rank, (doc_id, _) in enumerate(results, 1):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += 1.0 / (k + rank)
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def rerank(self, query: str, candidate_ids: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """Rerank candidates using cross-encoder"""
        if not self.use_rerank or len(candidate_ids) == 0:
            return [(idx, 0.0) for idx in candidate_ids[:top_k]]
        
        # Prepare pairs for reranking
        pairs = [[query, self.corpus[idx]] for idx in candidate_ids]
        
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        
        # Combine with indices and sort
        reranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]
    
    def retrieve(self, query: str, top_k: int = 5, method: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Main retrieval function
        
        Args:
            query: Query string
            top_k: Number of results to return
            method: "dense", "sparse", or "hybrid"
        
        Returns:
            List of retrieved documents with scores
        """
        if method == "dense":
            results = self.retrieve_dense(query, top_k * 2)
            candidate_ids = [idx for idx, _ in results]
        
        elif method == "sparse":
            results = self.retrieve_sparse(query, top_k * 2)
            candidate_ids = [idx for idx, _ in results]
        
        else:  # hybrid
            # Get results from both methods
            dense_results = self.retrieve_dense(query, top_k * 2)
            sparse_results = self.retrieve_sparse(query, top_k * 2)
            
            # Combine using RRF
            if dense_results and sparse_results:
                fused = self.reciprocal_rank_fusion([dense_results, sparse_results])
                candidate_ids = [idx for idx, _ in fused[:top_k * 2]]
            elif dense_results:
                candidate_ids = [idx for idx, _ in dense_results[:top_k * 2]]
            else:
                candidate_ids = [idx for idx, _ in sparse_results[:top_k * 2]]
        
        # Rerank if enabled
        if self.use_rerank and len(candidate_ids) > 0:
            reranked = self.rerank(query, candidate_ids, top_k)
            final_results = [
                {
                    **self.documents[idx],
                    'score': float(score),
                    'rank': rank + 1
                }
                for rank, (idx, score) in enumerate(reranked)
            ]
        else:
            final_results = [
                {
                    **self.documents[idx],
                    'score': 0.0,
                    'rank': rank + 1
                }
                for rank, idx in enumerate(candidate_ids[:top_k])
            ]
        
        return final_results


class QueryExpander:
    """Expand queries with synonyms and related terms"""
    
    def __init__(self):
        # Domain-specific synonym dictionary for OS and Networks
        self.synonyms = {
            # Operating Systems
            "process": ["task", "program execution", "job"],
            "thread": ["lightweight process", "execution unit"],
            "deadlock": ["circular wait", "resource deadlock"],
            "scheduling": ["CPU scheduling", "process scheduling"],
            "mutex": ["mutual exclusion", "lock"],
            "semaphore": ["synchronization primitive"],
            "memory": ["RAM", "main memory", "primary storage"],
            "virtual memory": ["paging", "virtual storage"],
            "page": ["page frame", "memory page"],
            "cache": ["buffer cache", "memory cache"],
            
            # Networks
            "TCP": ["transmission control protocol", "TCP/IP"],
            "UDP": ["user datagram protocol"],
            "IP": ["internet protocol", "IP address"],
            "router": ["gateway", "routing device"],
            "switch": ["network switch", "layer 2 switch"],
            "packet": ["data packet", "network packet"],
            "bandwidth": ["throughput", "network capacity"],
            "latency": ["delay", "network delay"],
            "protocol": ["communication protocol", "network protocol"],
            "DNS": ["domain name system", "name resolution"],
        }
    
    def expand_query(self, query: str, max_terms: int = 3) -> str:
        """
        Expand query with synonyms
        
        Args:
            query: Original query
            max_terms: Maximum number of expansion terms to add
        
        Returns:
            Expanded query
        """
        query_lower = query.lower()
        expanded_terms = []
        
        # Find matching terms and add synonyms
        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                # Add first few synonyms
                expanded_terms.extend(synonyms[:max_terms])
        
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms[:max_terms])}"
        
        return query
    
    def extract_key_terms(self, query: str) -> List[str]:
        """Extract key technical terms from query"""
        query_lower = query.lower()
        key_terms = []
        
        for term in self.synonyms.keys():
            if term in query_lower:
                key_terms.append(term)
        
        return key_terms
