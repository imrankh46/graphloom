"""
Evidence Retrieval for GraphLoom Phase 2.

This module provides evidence subgraph retrieval with:
- Dense triple scoring using unified multimodal embeddings
- Bounded graph expansion with degree capping and edge budgets
- Multi-hop retrieval support
"""

from typing import List, Tuple, Set, Optional, Dict, Any
from collections import deque
import numpy as np

from graphloom.core.data_structures import (
    Triple,
    MMKG,
    EvidenceSubgraph,
    RetrievalConfig,
)
from graphloom.phase2.multimodal_embedder import (
    BaseMultimodalEmbedder,
    create_multimodal_embedder,
)


class EvidenceRetriever:
    """
    Evidence subgraph retriever for GraphLoom.
    
    Retrieves a compact evidence subgraph from the MMKG using:
    1. Dense triple scoring with unified multimodal embeddings
    2. Bounded graph expansion (BFS with degree capping)
    3. Edge budget constraints
    
    Example:
        >>> retriever = EvidenceRetriever()
        >>> subgraph = retriever.retrieve(mmkg, image, question)
    """
    
    def __init__(
        self,
        embedder: Optional[BaseMultimodalEmbedder] = None,
        config: Optional[RetrievalConfig] = None,
        embedder_type: str = "mock",
        **kwargs: Any,
    ):
        """
        Initialize the evidence retriever.
        
        Args:
            embedder: Pre-configured multimodal embedder (optional)
            config: Retrieval configuration (optional)
            embedder_type: Type of embedder to create if not provided
            **kwargs: Additional arguments for embedder creation
        """
        self.embedder = embedder or create_multimodal_embedder(
            embedder_type=embedder_type,
            **kwargs.get("embedder_kwargs", {}),
        )
        self.config = config or RetrievalConfig()
        self._triple_embeddings_cache: Dict[Triple, np.ndarray] = {}
    
    def compute_triple_embedding(self, triple: Triple) -> np.ndarray:
        """
        Compute or retrieve cached embedding for a triple.
        
        Args:
            triple: The triple to embed
            
        Returns:
            Embedding vector for the triple
        """
        if triple in self._triple_embeddings_cache:
            return self._triple_embeddings_cache[triple]
        
        if triple.embedding is not None:
            self._triple_embeddings_cache[triple] = triple.embedding
            return triple.embedding
        
        text = triple.to_text()
        embedding = self.embedder.embed_text(text)
        self._triple_embeddings_cache[triple] = embedding
        triple.embedding = embedding
        
        return embedding
    
    def compute_triple_embeddings_batch(
        self,
        triples: List[Triple],
    ) -> np.ndarray:
        """
        Compute embeddings for multiple triples in batch.
        
        Args:
            triples: List of triples to embed
            
        Returns:
            Array of embedding vectors (N x embedding_dim)
        """
        texts = [t.to_text() for t in triples]
        embeddings = self.embedder.embed_texts(texts)
        
        for triple, embedding in zip(triples, embeddings):
            self._triple_embeddings_cache[triple] = embedding
            triple.embedding = embedding
        
        return embeddings
    
    def score_triples(
        self,
        query_embedding: np.ndarray,
        triples: List[Triple],
    ) -> List[Tuple[Triple, float]]:
        """
        Score triples by similarity to query embedding.
        
        Args:
            query_embedding: The multimodal query embedding
            triples: List of triples to score
            
        Returns:
            List of (triple, score) tuples sorted by score descending
        """
        if not triples:
            return []
        
        uncached_triples = [
            t for t in triples
            if t not in self._triple_embeddings_cache and t.embedding is None
        ]
        
        if uncached_triples:
            self.compute_triple_embeddings_batch(uncached_triples)
        
        scored = []
        for triple in triples:
            embedding = self._triple_embeddings_cache.get(triple, triple.embedding)
            if embedding is not None:
                score = float(np.dot(query_embedding, embedding))
                scored.append((triple, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def select_top_k(
        self,
        scored_triples: List[Tuple[Triple, float]],
        k: Optional[int] = None,
    ) -> List[Triple]:
        """
        Select top-k triples by score.
        
        Args:
            scored_triples: List of (triple, score) tuples
            k: Number of triples to select (default: config.top_k)
            
        Returns:
            List of top-k triples
        """
        k = k or self.config.top_k
        return [t for t, _ in scored_triples[:k]]
    
    def bounded_graph_expansion(
        self,
        mmkg: MMKG,
        seed_triples: List[Triple],
        query_embedding: np.ndarray,
    ) -> List[Triple]:
        """
        Perform bounded BFS expansion from seed triples.
        
        Starting from the endpoints of seed triples, performs breadth-first
        traversal with:
        - Degree capping: At most max_degree edges per node
        - Edge budget: Total edges capped at max_edges
        - Hop limit: Maximum max_hops from seed nodes
        
        Args:
            mmkg: The MMKG to expand from
            seed_triples: Initial seed triples
            query_embedding: Query embedding for scoring expansion candidates
            
        Returns:
            List of expanded triples (excluding seeds)
        """
        seed_entities: Set[str] = set()
        for triple in seed_triples:
            seed_entities.add(triple.subject)
            seed_entities.add(triple.object)
        
        visited_triples: Set[Triple] = set(seed_triples)
        expanded_triples: List[Triple] = []
        
        queue: deque = deque()
        for entity in seed_entities:
            queue.append((entity, 0))
        
        visited_entities: Set[str] = set(seed_entities)
        
        while queue and len(expanded_triples) < (self.config.max_edges - len(seed_triples)):
            entity, hop = queue.popleft()
            
            if hop >= self.config.max_hops:
                continue
            
            neighbors = mmkg.get_neighbors(entity, max_degree=-1)
            
            candidate_triples = [t for t in neighbors if t not in visited_triples]
            
            if candidate_triples:
                scored = self.score_triples(query_embedding, candidate_triples)
                top_neighbors = scored[:self.config.max_degree]
                
                for triple, score in top_neighbors:
                    if len(expanded_triples) >= (self.config.max_edges - len(seed_triples)):
                        break
                    
                    if score < self.config.similarity_threshold:
                        continue
                    
                    visited_triples.add(triple)
                    expanded_triples.append(triple)
                    
                    next_entity = (
                        triple.object if triple.subject == entity else triple.subject
                    )
                    if next_entity not in visited_entities:
                        visited_entities.add(next_entity)
                        queue.append((next_entity, hop + 1))
        
        return expanded_triples
    
    def retrieve(
        self,
        mmkg: MMKG,
        image: Any,
        question: str,
        extracted_triples: Optional[List[Triple]] = None,
    ) -> EvidenceSubgraph:
        """
        Retrieve a compact evidence subgraph from the MMKG.
        
        Steps:
        1. Compute multimodal query embedding
        2. Score all MMKG triples against query
        3. Select top-k seed triples
        4. Perform bounded graph expansion
        5. Return evidence subgraph
        
        Args:
            mmkg: The MMKG to retrieve from
            image: Image for multimodal query
            question: Question text
            extracted_triples: Optional newly extracted triples to prioritize
            
        Returns:
            EvidenceSubgraph containing seed and expanded triples
        """
        query_embedding = self.embedder.embed_query(image, question)
        
        all_triples = mmkg.get_all_triples()
        
        if extracted_triples:
            for triple in extracted_triples:
                if triple not in all_triples:
                    all_triples.append(triple)
        
        scored_triples = self.score_triples(query_embedding, all_triples)
        
        seed_triples = self.select_top_k(scored_triples)
        
        expanded_triples = self.bounded_graph_expansion(
            mmkg, seed_triples, query_embedding
        )
        
        subgraph_mmkg = MMKG()
        for triple in seed_triples + expanded_triples:
            subgraph_mmkg.add(triple)
        
        return EvidenceSubgraph(
            seed_triples=seed_triples,
            expanded_triples=expanded_triples,
            mmkg=subgraph_mmkg,
            query_embedding=query_embedding,
        )


def unified_similarity_score(
    query_embedding: np.ndarray,
    triple_embedding: np.ndarray,
) -> float:
    """
    Compute unified similarity score between query and triple embeddings.
    
    Uses cosine similarity (dot product of normalized vectors).
    
    Args:
        query_embedding: Query embedding vector
        triple_embedding: Triple embedding vector
        
    Returns:
        Similarity score in [-1, 1]
    """
    return float(np.dot(query_embedding, triple_embedding))


def multimodal_retrieval(
    mmkg: MMKG,
    query_embedding: np.ndarray,
    embedder: BaseMultimodalEmbedder,
    top_k: int = 15,
) -> List[Tuple[Triple, float]]:
    """
    Perform multimodal retrieval from MMKG.
    
    Convenience function matching pseudocode signature.
    
    Args:
        mmkg: The MMKG to retrieve from
        query_embedding: Multimodal query embedding
        embedder: Embedder for computing triple embeddings
        top_k: Number of triples to retrieve
        
    Returns:
        List of (triple, score) tuples
    """
    retriever = EvidenceRetriever(embedder=embedder)
    all_triples = mmkg.get_all_triples()
    scored = retriever.score_triples(query_embedding, all_triples)
    return scored[:top_k]
