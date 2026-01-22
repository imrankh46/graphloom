"""
Phase 2 Pipeline: Evidence Subgraph Retrieval.

This module orchestrates the Phase 2 pipeline:
1. Compute multimodal query embedding
2. Retrieve candidate triples from MMKG
3. Score triples with unified similarity
4. Select top-k triples
5. Perform bounded graph expansion
"""

from typing import Union, List, Tuple, Optional, Any
from pathlib import Path
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
from graphloom.phase2.retrieval import EvidenceRetriever


class Phase2Pipeline:
    """
    Phase 2: Evidence Subgraph Retrieval Pipeline.
    
    This pipeline handles:
    1. Multimodal query embedding computation
    2. Dense triple scoring against MMKG
    3. Top-k triple selection
    4. Bounded graph expansion
    
    Example:
        >>> pipeline = Phase2Pipeline()
        >>> subgraph = pipeline.process(mmkg, extracted_triples, image, question)
    """
    
    def __init__(
        self,
        embedder: Optional[BaseMultimodalEmbedder] = None,
        retriever: Optional[EvidenceRetriever] = None,
        config: Optional[RetrievalConfig] = None,
        embedder_type: str = "mock",
        **kwargs: Any,
    ):
        """
        Initialize the Phase 2 pipeline.
        
        Args:
            embedder: Pre-configured multimodal embedder (optional)
            retriever: Pre-configured evidence retriever (optional)
            config: Retrieval configuration (optional)
            embedder_type: Type of embedder to create if not provided
            **kwargs: Additional arguments for component creation
        """
        self.config = config or RetrievalConfig()
        
        self.embedder = embedder or create_multimodal_embedder(
            embedder_type=embedder_type,
            **kwargs.get("embedder_kwargs", {}),
        )
        
        self.retriever = retriever or EvidenceRetriever(
            embedder=self.embedder,
            config=self.config,
        )
    
    def compute_query_embedding(
        self,
        image: Union[str, Path, np.ndarray, None],
        question: str,
    ) -> np.ndarray:
        """
        Compute multimodal query embedding.
        
        Args:
            image: Image path, URL, numpy array, or None
            question: Question text
            
        Returns:
            Query embedding vector
        """
        return self.embedder.embed_query(image, question)
    
    def retrieve_candidates(
        self,
        mmkg: MMKG,
        query_embedding: np.ndarray,
        extracted_triples: Optional[List[Triple]] = None,
    ) -> List[Tuple[Triple, float]]:
        """
        Retrieve and score candidate triples from MMKG.
        
        Args:
            mmkg: The MMKG to retrieve from
            query_embedding: Multimodal query embedding
            extracted_triples: Optional newly extracted triples to include
            
        Returns:
            List of (triple, score) tuples sorted by score
        """
        all_triples = mmkg.get_all_triples()
        
        if extracted_triples:
            triple_set = set(all_triples)
            for triple in extracted_triples:
                if triple not in triple_set:
                    all_triples.append(triple)
        
        return self.retriever.score_triples(query_embedding, all_triples)
    
    def select_top_triples(
        self,
        scored_triples: List[Tuple[Triple, float]],
        k: Optional[int] = None,
    ) -> List[Triple]:
        """
        Select top-k triples by score.
        
        Args:
            scored_triples: List of (triple, score) tuples
            k: Number of triples to select
            
        Returns:
            List of top-k triples
        """
        k = k or self.config.top_k
        return [t for t, _ in scored_triples[:k]]
    
    def expand_subgraph(
        self,
        mmkg: MMKG,
        seed_triples: List[Triple],
        query_embedding: np.ndarray,
    ) -> List[Triple]:
        """
        Perform bounded graph expansion from seed triples.
        
        Args:
            mmkg: The MMKG to expand from
            seed_triples: Initial seed triples
            query_embedding: Query embedding for scoring
            
        Returns:
            List of expanded triples
        """
        return self.retriever.bounded_graph_expansion(
            mmkg, seed_triples, query_embedding
        )
    
    def process(
        self,
        mmkg: MMKG,
        extracted_triples: List[Triple],
        image: Union[str, Path, np.ndarray, None],
        question: str,
    ) -> EvidenceSubgraph:
        """
        Execute the full Phase 2 pipeline.
        
        Steps:
        1. Compute multimodal query embedding
        2. Retrieve and score candidate triples
        3. Select top-k seed triples
        4. Perform bounded graph expansion
        5. Build and return evidence subgraph
        
        Args:
            mmkg: The MMKG to retrieve from
            extracted_triples: Triples extracted in Phase 1
            image: Image for multimodal query
            question: Question text
            
        Returns:
            EvidenceSubgraph containing compact evidence
        """
        query_embedding = self.compute_query_embedding(image, question)
        
        scored_triples = self.retrieve_candidates(
            mmkg, query_embedding, extracted_triples
        )
        
        seed_triples = self.select_top_triples(scored_triples)
        
        expanded_triples = self.expand_subgraph(
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


def phase_2_simplified(
    mmkg: MMKG,
    extracted_triples: List[Triple],
    image: Any,
    question: str,
    pipeline: Optional[Phase2Pipeline] = None,
    **kwargs: Any,
) -> EvidenceSubgraph:
    """
    Convenience function for Phase 2 processing.
    
    Matches the pseudocode signature:
    function PHASE_2_SIMPLIFIED(MMKG, extracted_triples, image I, question Q):
        ...
        return compact_subgraph
    
    Args:
        mmkg: The MMKG to retrieve from
        extracted_triples: Triples extracted in Phase 1
        image: Image for multimodal query
        question: Question text
        pipeline: Optional pre-configured pipeline
        **kwargs: Arguments for pipeline creation
        
    Returns:
        EvidenceSubgraph containing compact evidence
    """
    if pipeline is None:
        pipeline = Phase2Pipeline(**kwargs)
    
    return pipeline.process(mmkg, extracted_triples, image, question)
