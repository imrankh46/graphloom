"""
HieraSlot Encoder for GraphLoom Phase 3.

This module provides hierarchical slot memory encoding for evidence triples.
Each triple is converted into entity and relation slots for fine-grained attention.
"""

from typing import List, Optional, Any, Tuple
import numpy as np

from graphloom.core.data_structures import (
    Triple,
    HieraSlot,
    SlotBank,
    SlotType,
    EvidenceSubgraph,
)
from graphloom.phase2.multimodal_embedder import (
    BaseMultimodalEmbedder,
    create_multimodal_embedder,
)


class HieraSlotEncoder:
    """
    Hierarchical Slot Memory Encoder.
    
    Converts retrieved triples into structured slot memories with two levels:
    - Entity slots: Encode entity-specific information (key=subject, value=relation+object)
    - Relation slots: Encode relational semantics (key=relation, value=subject+object)
    
    This hierarchical organization allows fine-grained attention to either
    entity attributes or relational facts during decoding.
    
    Example:
        >>> encoder = HieraSlotEncoder()
        >>> slot_bank = encoder.encode(evidence_subgraph)
    """
    
    def __init__(
        self,
        embedder: Optional[BaseMultimodalEmbedder] = None,
        embedder_type: str = "mock",
        hidden_dim: int = 4096,
        **kwargs: Any,
    ):
        """
        Initialize the HieraSlot encoder.
        
        Args:
            embedder: Pre-configured embedder for text encoding
            embedder_type: Type of embedder to create if not provided
            hidden_dim: Hidden dimension for slot projections (matches decoder)
            **kwargs: Additional arguments for embedder creation
        """
        self.embedder = embedder or create_multimodal_embedder(
            embedder_type=embedder_type,
            **kwargs.get("embedder_kwargs", {}),
        )
        self.hidden_dim = hidden_dim
        
        self._projection_matrix: Optional[np.ndarray] = None
        self._projection_bias: Optional[np.ndarray] = None
    
    def _init_projection(self, input_dim: int) -> None:
        """Initialize projection matrix for slot encoding."""
        if self._projection_matrix is not None:
            return
        
        np.random.seed(42)
        scale = np.sqrt(2.0 / (input_dim + self.hidden_dim))
        self._projection_matrix = np.random.randn(input_dim, self.hidden_dim) * scale
        self._projection_bias = np.zeros(self.hidden_dim)
    
    def _project_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding to hidden dimension."""
        self._init_projection(embedding.shape[-1])
        projected = np.dot(embedding, self._projection_matrix) + self._projection_bias
        return projected
    
    def encode_triple(
        self,
        triple: Triple,
        retrieval_score: float = 0.0,
    ) -> Tuple[HieraSlot, HieraSlot]:
        """
        Encode a single triple into entity and relation slots.
        
        Entity slot:
            key = Embed(subject)
            value = Embed(relation + object)
        
        Relation slot:
            key = Embed(relation)
            value = Embed(subject + object)
        
        Args:
            triple: The triple to encode
            retrieval_score: Original retrieval relevance score
            
        Returns:
            Tuple of (entity_slot, relation_slot)
        """
        subject_emb = self.embedder.embed_text(triple.subject)
        relation_emb = self.embedder.embed_text(triple.relation)
        object_emb = self.embedder.embed_text(triple.object)
        
        relation_object_text = f"{triple.relation} {triple.object}"
        relation_object_emb = self.embedder.embed_text(relation_object_text)
        
        subject_object_text = f"{triple.subject} {triple.object}"
        subject_object_emb = self.embedder.embed_text(subject_object_text)
        
        key_entity = self._project_embedding(subject_emb)
        value_entity = self._project_embedding(relation_object_emb)
        
        key_relation = self._project_embedding(relation_emb)
        value_relation = self._project_embedding(subject_object_emb)
        
        reliability_score = triple.confidence * 0.5 + 0.5
        
        entity_slot = HieraSlot(
            slot_type=SlotType.ENTITY,
            key=key_entity,
            value=value_entity,
            source_triple=triple,
            reliability_score=reliability_score,
            retrieval_score=retrieval_score,
        )
        
        relation_slot = HieraSlot(
            slot_type=SlotType.RELATION,
            key=key_relation,
            value=value_relation,
            source_triple=triple,
            reliability_score=reliability_score,
            retrieval_score=retrieval_score,
        )
        
        return entity_slot, relation_slot
    
    def encode(
        self,
        evidence_subgraph: EvidenceSubgraph,
        scored_triples: Optional[List[Tuple[Triple, float]]] = None,
    ) -> SlotBank:
        """
        Encode all triples in an evidence subgraph into a SlotBank.
        
        Args:
            evidence_subgraph: The evidence subgraph to encode
            scored_triples: Optional list of (triple, score) for retrieval scores
            
        Returns:
            SlotBank containing all entity and relation slots
        """
        slot_bank = SlotBank()
        
        score_map = {}
        if scored_triples:
            score_map = {t: s for t, s in scored_triples}
        
        all_triples = evidence_subgraph.all_triples
        
        for triple in all_triples:
            retrieval_score = score_map.get(triple, 0.5)
            entity_slot, relation_slot = self.encode_triple(triple, retrieval_score)
            slot_bank.entity_slots.append(entity_slot)
            slot_bank.relation_slots.append(relation_slot)
        
        return slot_bank
    
    def encode_triples(
        self,
        triples: List[Triple],
        retrieval_scores: Optional[List[float]] = None,
    ) -> SlotBank:
        """
        Encode a list of triples into a SlotBank.
        
        Args:
            triples: List of triples to encode
            retrieval_scores: Optional retrieval scores for each triple
            
        Returns:
            SlotBank containing all entity and relation slots
        """
        slot_bank = SlotBank()
        
        if retrieval_scores is None:
            retrieval_scores = [0.5] * len(triples)
        
        for triple, score in zip(triples, retrieval_scores):
            entity_slot, relation_slot = self.encode_triple(triple, score)
            slot_bank.entity_slots.append(entity_slot)
            slot_bank.relation_slots.append(relation_slot)
        
        return slot_bank


def hieraslot_encode(
    compact_subgraph: EvidenceSubgraph,
    encoder: Optional[HieraSlotEncoder] = None,
    **kwargs: Any,
) -> SlotBank:
    """
    Convenience function for HieraSlot encoding.
    
    Matches the pseudocode:
    slots = HieraSlot_encode(compact_subgraph)
    
    Args:
        compact_subgraph: Evidence subgraph to encode
        encoder: Optional pre-configured encoder
        **kwargs: Arguments for encoder creation
        
    Returns:
        SlotBank containing encoded slots
    """
    if encoder is None:
        encoder = HieraSlotEncoder(**kwargs)
    
    return encoder.encode(compact_subgraph)
