"""
Core data structures for GraphLoom.

This module defines the fundamental data types used throughout the GraphLoom pipeline:
- Triple: Knowledge graph triple (subject, relation, object)
- MMKG: Multimodal Knowledge Graph
- HieraSlot: Hierarchical slot memory for evidence encoding
- SlotBank: Collection of HieraSlot memories
- EvidenceSubgraph: Retrieved compact evidence subgraph
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Tuple, Any
from collections import defaultdict
import numpy as np
from enum import Enum


class SlotType(Enum):
    """Type of HieraSlot memory."""
    ENTITY = "entity"
    RELATION = "relation"


@dataclass
class Triple:
    """
    A knowledge graph triple representing a factual relation.
    
    Attributes:
        subject: The subject entity of the triple
        relation: The relation/predicate connecting subject and object
        object: The object entity of the triple
        confidence: Extraction confidence score from REBEL [0, 1]
        source: Origin of the triple (e.g., 'rebel', 'dataset')
        embedding: Optional cached embedding vector
        metadata: Additional metadata (e.g., image_id, question_id)
    """
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    source: str = "dataset"
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash((self.subject, self.relation, self.object))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Triple):
            return False
        return (
            self.subject == other.subject
            and self.relation == other.relation
            and self.object == other.object
        )
    
    def to_text(self) -> str:
        """Convert triple to natural language verbalization."""
        return f"{self.subject} {self.relation} {self.object}"
    
    def to_tuple(self) -> Tuple[str, str, str]:
        """Return triple as a tuple."""
        return (self.subject, self.relation, self.object)


@dataclass
class HieraSlot:
    """
    Hierarchical slot memory for evidence encoding.
    
    Each triple is converted into two specialized slots:
    - Entity slot: Encodes entity-specific information
    - Relation slot: Encodes relational semantics
    
    Attributes:
        slot_type: Type of slot (ENTITY or RELATION)
        key: Key vector for attention computation
        value: Value vector containing slot content
        source_triple: The original triple this slot was derived from
        reliability_score: Calibrated reliability score [0, 1]
        retrieval_score: Original retrieval relevance score
    """
    slot_type: SlotType
    key: np.ndarray
    value: np.ndarray
    source_triple: Triple
    reliability_score: float = 0.5
    retrieval_score: float = 0.0
    
    @property
    def triple_text(self) -> str:
        """Get text representation of source triple."""
        return self.source_triple.to_text()


@dataclass
class SlotBank:
    """
    Collection of HieraSlot memories for a retrieved evidence subgraph.
    
    Attributes:
        entity_slots: List of entity-type slots
        relation_slots: List of relation-type slots
    """
    entity_slots: List[HieraSlot] = field(default_factory=list)
    relation_slots: List[HieraSlot] = field(default_factory=list)
    
    def add_slots_from_triple(
        self,
        triple: Triple,
        key_entity: np.ndarray,
        value_entity: np.ndarray,
        key_relation: np.ndarray,
        value_relation: np.ndarray,
        reliability_score: float = 0.5,
        retrieval_score: float = 0.0,
    ) -> None:
        """Add entity and relation slots derived from a triple."""
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
        self.entity_slots.append(entity_slot)
        self.relation_slots.append(relation_slot)
    
    @property
    def all_slots(self) -> List[HieraSlot]:
        """Get all slots (entity + relation)."""
        return self.entity_slots + self.relation_slots
    
    def __len__(self) -> int:
        return len(self.entity_slots) + len(self.relation_slots)


class MMKG:
    """
    Multimodal Knowledge Graph.
    
    A graph structure storing triples with efficient lookup by entity.
    Supports adding triples, querying neighbors, and subgraph extraction.
    
    Attributes:
        triples: Set of all triples in the graph
        entity_to_triples: Index mapping entities to their triples
        confidence_threshold: Minimum confidence for triple inclusion
    """
    
    def __init__(self, confidence_threshold: float = 0.0):
        """
        Initialize an empty MMKG.
        
        Args:
            confidence_threshold: Minimum confidence score for adding triples
        """
        self.triples: Set[Triple] = set()
        self.entity_to_triples: Dict[str, Set[Triple]] = defaultdict(set)
        self.confidence_threshold = confidence_threshold
        self._embeddings_cache: Dict[Triple, np.ndarray] = {}
    
    def add(self, triple: Triple) -> bool:
        """
        Add a triple to the MMKG.
        
        Args:
            triple: The triple to add
            
        Returns:
            True if triple was added, False if rejected (low confidence or duplicate)
        """
        if triple.confidence < self.confidence_threshold:
            return False
        
        if triple in self.triples:
            return False
        
        self.triples.add(triple)
        self.entity_to_triples[triple.subject].add(triple)
        self.entity_to_triples[triple.object].add(triple)
        
        if triple.embedding is not None:
            self._embeddings_cache[triple] = triple.embedding
        
        return True
    
    def add_batch(self, triples: List[Triple]) -> int:
        """
        Add multiple triples to the MMKG.
        
        Args:
            triples: List of triples to add
            
        Returns:
            Number of triples successfully added
        """
        added = 0
        for triple in triples:
            if self.add(triple):
                added += 1
        return added
    
    def get_neighbors(self, entity: str, max_degree: int = -1) -> List[Triple]:
        """
        Get all triples connected to an entity.
        
        Args:
            entity: The entity to find neighbors for
            max_degree: Maximum number of neighbors to return (-1 for all)
            
        Returns:
            List of triples containing the entity
        """
        neighbors = list(self.entity_to_triples.get(entity, set()))
        if max_degree > 0 and len(neighbors) > max_degree:
            neighbors = sorted(neighbors, key=lambda t: t.confidence, reverse=True)
            neighbors = neighbors[:max_degree]
        return neighbors
    
    def get_entities(self) -> Set[str]:
        """Get all unique entities in the graph."""
        return set(self.entity_to_triples.keys())
    
    def get_all_triples(self) -> List[Triple]:
        """Get all triples as a list."""
        return list(self.triples)
    
    def subgraph_from_entities(self, entities: Set[str]) -> "MMKG":
        """
        Extract a subgraph containing only the specified entities.
        
        Args:
            entities: Set of entities to include
            
        Returns:
            New MMKG containing only triples with both endpoints in entities
        """
        subgraph = MMKG(confidence_threshold=self.confidence_threshold)
        for triple in self.triples:
            if triple.subject in entities and triple.object in entities:
                subgraph.add(triple)
        return subgraph
    
    def __len__(self) -> int:
        return len(self.triples)
    
    def __contains__(self, triple: Triple) -> bool:
        return triple in self.triples
    
    def __iter__(self):
        return iter(self.triples)


@dataclass
class EvidenceSubgraph:
    """
    A compact evidence subgraph retrieved for answering a question.
    
    Attributes:
        seed_triples: Initial top-k retrieved triples
        expanded_triples: Additional triples from bounded expansion
        mmkg: The subgraph as an MMKG structure
        query_embedding: The multimodal query embedding used for retrieval
    """
    seed_triples: List[Triple]
    expanded_triples: List[Triple]
    mmkg: MMKG
    query_embedding: Optional[np.ndarray] = None
    
    @property
    def all_triples(self) -> List[Triple]:
        """Get all triples in the evidence subgraph."""
        return self.seed_triples + self.expanded_triples
    
    def to_text(self) -> str:
        """Convert evidence subgraph to text for prompting."""
        lines = []
        for i, triple in enumerate(self.all_triples, 1):
            lines.append(f"{i}. {triple.to_text()}")
        return "\n".join(lines)


@dataclass
class RetrievalConfig:
    """Configuration for evidence subgraph retrieval."""
    top_k: int = 15
    max_hops: int = 2
    max_degree: int = 5
    max_edges: int = 50
    similarity_threshold: float = 0.3
    
    
@dataclass
class RouterConfig:
    """Configuration for reliability-calibrated slot routing."""
    top_k_slots: int = 4
    gate_threshold: float = 0.3
    reliability_weight: float = 2.0
    reliability_bias: float = -1.0
    reliability_clamp_epsilon: float = 0.1
