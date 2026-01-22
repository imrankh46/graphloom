"""
Phase 3 Pipeline: Controlled Reasoning.

This module orchestrates the Phase 3 pipeline:
1. HieraSlot encoding of evidence subgraph
2. Reliability-calibrated slot routing
3. KG-JSA++ joint attention
4. Faithful answer generation
"""

from typing import Union, List, Optional, Any
from pathlib import Path
import numpy as np

from graphloom.core.data_structures import (
    Triple,
    MMKG,
    EvidenceSubgraph,
    HieraSlot,
    SlotBank,
    RouterConfig,
)
from graphloom.phase3.hieraslot import HieraSlotEncoder
from graphloom.phase3.router import (
    ReliabilityCalibratedRouter,
    RoutedSlots,
)
from graphloom.phase3.kg_jsa import KGJSAPlusPlus, PrefixMemory
from graphloom.phase3.answer_generator import (
    BaseAnswerGenerator,
    create_answer_generator,
    GeneratedAnswer,
)


class Phase3Pipeline:
    """
    Phase 3: Controlled Reasoning Pipeline.
    
    This pipeline handles:
    1. HieraSlot encoding - Convert evidence triples to slot memories
    2. Reliability-calibrated routing - Select top-k slots per step
    3. KG-JSA++ attention - Joint graph-sequence attention
    4. Faithful answer generation - Generate grounded answers
    
    Example:
        >>> pipeline = Phase3Pipeline()
        >>> answer = pipeline.process(evidence_subgraph, image, question)
    """
    
    def __init__(
        self,
        slot_encoder: Optional[HieraSlotEncoder] = None,
        router: Optional[ReliabilityCalibratedRouter] = None,
        kg_jsa: Optional[KGJSAPlusPlus] = None,
        answer_generator: Optional[BaseAnswerGenerator] = None,
        router_config: Optional[RouterConfig] = None,
        hidden_dim: int = 4096,
        generator_type: str = "mock",
        embedder_type: str = "mock",
        **kwargs: Any,
    ):
        """
        Initialize the Phase 3 pipeline.
        
        Args:
            slot_encoder: Pre-configured HieraSlot encoder (optional)
            router: Pre-configured slot router (optional)
            kg_jsa: Pre-configured KG-JSA++ module (optional)
            answer_generator: Pre-configured answer generator (optional)
            router_config: Router configuration (optional)
            hidden_dim: Hidden dimension for all components
            generator_type: Type of answer generator
            embedder_type: Type of embedder for slot encoding
            **kwargs: Additional arguments for component creation
        """
        self.hidden_dim = hidden_dim
        self.router_config = router_config or RouterConfig()
        
        self.slot_encoder = slot_encoder or HieraSlotEncoder(
            embedder_type=embedder_type,
            hidden_dim=hidden_dim,
            **kwargs.get("encoder_kwargs", {}),
        )
        
        self.router = router or ReliabilityCalibratedRouter(
            config=self.router_config,
            hidden_dim=hidden_dim,
        )
        
        self.kg_jsa = kg_jsa or KGJSAPlusPlus(
            hidden_dim=hidden_dim,
            **kwargs.get("kg_jsa_kwargs", {}),
        )
        
        self.answer_generator = answer_generator or create_answer_generator(
            generator_type=generator_type,
            **kwargs.get("generator_kwargs", {}),
        )
    
    def encode_slots(
        self,
        evidence_subgraph: EvidenceSubgraph,
    ) -> SlotBank:
        """
        Encode evidence subgraph into HieraSlot memories.
        
        Args:
            evidence_subgraph: Evidence subgraph from Phase 2
            
        Returns:
            SlotBank containing entity and relation slots
        """
        return self.slot_encoder.encode(evidence_subgraph)
    
    def route_slots(
        self,
        slot_bank: SlotBank,
        query_hidden: np.ndarray,
        visual_context: Optional[np.ndarray] = None,
    ) -> RoutedSlots:
        """
        Route slots to select top-k active slots.
        
        Args:
            slot_bank: SlotBank containing all slots
            query_hidden: Query hidden state
            visual_context: Optional visual context
            
        Returns:
            RoutedSlots containing active slots
        """
        return self.router.route(slot_bank, query_hidden, visual_context)
    
    def compute_joint_context(
        self,
        question: str,
        evidence_subgraph: EvidenceSubgraph,
        routed_slots: RoutedSlots,
    ) -> np.ndarray:
        """
        Compute joint context using KG-JSA++ attention.
        
        Args:
            question: Question text
            evidence_subgraph: Evidence subgraph
            routed_slots: Routed slots from router
            
        Returns:
            Joint context vector
        """
        evidence_text = evidence_subgraph.to_text()
        prefix_memory = self.kg_jsa.create_prefix_memory(question, evidence_text)
        
        np.random.seed(hash(question) % (2**32))
        query = np.random.randn(self.hidden_dim)
        query = query / np.linalg.norm(query)
        
        output = self.kg_jsa.forward(
            query=query,
            prefix_memory=prefix_memory,
            routed_slots=routed_slots.active_slots,
        )
        
        return output.output
    
    def generate_answer(
        self,
        question: str,
        joint_context: np.ndarray,
        evidence_subgraph: EvidenceSubgraph,
    ) -> GeneratedAnswer:
        """
        Generate faithful answer using joint context and evidence.
        
        Args:
            question: Question text
            joint_context: Joint context from KG-JSA++
            evidence_subgraph: Evidence subgraph for grounding
            
        Returns:
            GeneratedAnswer containing answer and metadata
        """
        return self.answer_generator.generate(
            question=question,
            joint_context=joint_context,
            evidence_subgraph=evidence_subgraph,
        )
    
    def process(
        self,
        evidence_subgraph: EvidenceSubgraph,
        image: Any,
        question: str,
    ) -> str:
        """
        Execute the full Phase 3 pipeline.
        
        Steps:
        1. Encode evidence subgraph into HieraSlot memories
        2. Route slots using reliability-calibrated router
        3. Compute joint context with KG-JSA++ attention
        4. Generate faithful answer
        
        Args:
            evidence_subgraph: Evidence subgraph from Phase 2
            image: Input image
            question: Question text
            
        Returns:
            Generated answer string
        """
        slot_bank = self.encode_slots(evidence_subgraph)
        
        np.random.seed(hash(question) % (2**32))
        query_hidden = np.random.randn(self.hidden_dim)
        query_hidden = query_hidden / np.linalg.norm(query_hidden)
        
        routed_slots = self.route_slots(slot_bank, query_hidden)
        
        joint_context = self.compute_joint_context(
            question, evidence_subgraph, routed_slots
        )
        
        result = self.generate_answer(question, joint_context, evidence_subgraph)
        
        return result.answer


def phase_3_controlled(
    compact_subgraph: EvidenceSubgraph,
    image: Any,
    question: str,
    pipeline: Optional[Phase3Pipeline] = None,
    **kwargs: Any,
) -> str:
    """
    Convenience function for Phase 3 processing.
    
    Matches the pseudocode:
    function PHASE_3_CONTROLLED(compact_subgraph, image I, question Q):
        slots = HieraSlot_encode(compact_subgraph)
        routed = reliability_calibrated_router(slots, Q)
        top_slots = select_top_k_slots(routed, k = K_SLOTS)
        joint_context = KG_JSA_pp_attention(
            image = I,
            question = Q,
            slots = top_slots
        )
        final_answer = generate_faithful_answer(
            question = Q,
            joint_context = joint_context,
            evidence_subgraph = compact_subgraph
        )
        return final_answer
    
    Args:
        compact_subgraph: Evidence subgraph from Phase 2
        image: Input image
        question: Question text
        pipeline: Optional pre-configured pipeline
        **kwargs: Arguments for pipeline creation
        
    Returns:
        Generated answer string
    """
    if pipeline is None:
        pipeline = Phase3Pipeline(**kwargs)
    
    return pipeline.process(compact_subgraph, image, question)
