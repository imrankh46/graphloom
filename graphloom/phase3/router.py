"""
Reliability-Calibrated Slot Router for GraphLoom Phase 3.

This module provides dynamic slot routing that activates only a top-k subset
of HieraSlot memories at each decoding step based on expected utility.
"""

from typing import List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from graphloom.core.data_structures import (
    HieraSlot,
    SlotBank,
    RouterConfig,
)


@dataclass
class RoutedSlots:
    """Result of slot routing for a decoding step."""
    active_slots: List[HieraSlot]
    gate_probabilities: np.ndarray
    utility_scores: np.ndarray
    active_indices: List[int]


class ReliabilityCalibratedRouter:
    """
    Reliability-Calibrated Slot Router.
    
    At each decoding step, activates only a top-k subset of HieraSlot memories
    based on:
    - Semantic alignment between query and slot keys
    - Visual context (when available)
    - Slot features (retrieval score, graph statistics)
    - Calibrated reliability scores
    
    The router uses evidence-aware features to predict slot utility while
    maintaining efficiency through sparse activation.
    
    Example:
        >>> router = ReliabilityCalibratedRouter()
        >>> routed = router.route(slot_bank, query_hidden, visual_context)
    """
    
    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        hidden_dim: int = 4096,
        **kwargs: Any,
    ):
        """
        Initialize the router.
        
        Args:
            config: Router configuration
            hidden_dim: Hidden dimension matching decoder
            **kwargs: Additional arguments
        """
        self.config = config or RouterConfig()
        self.hidden_dim = hidden_dim
        
        self._query_projection: Optional[np.ndarray] = None
        self._feature_projection: Optional[np.ndarray] = None
        self._visual_projection: Optional[np.ndarray] = None
    
    def _init_projections(self) -> None:
        """Initialize projection matrices."""
        if self._query_projection is not None:
            return
        
        np.random.seed(42)
        scale = np.sqrt(2.0 / self.hidden_dim)
        
        self._query_projection = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self._feature_projection = np.random.randn(4, 1) * 0.1
        self._visual_projection = np.random.randn(self.hidden_dim, 1) * scale
    
    def compute_utility_scores(
        self,
        slots: List[HieraSlot],
        query_hidden: np.ndarray,
        visual_context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute utility scores for all slots.
        
        Utility combines:
        - Semantic alignment: q^T * k (query-key dot product)
        - Visual context: w_v^T * v_t (visual projection)
        - Slot features: u_f^T * phi(f_p) (feature projection)
        - Reliability bias: beta * logit(c_p) (calibrated reliability)
        
        Args:
            slots: List of HieraSlot memories
            query_hidden: Query hidden state from decoder
            visual_context: Optional pooled visual context
            
        Returns:
            Array of utility scores for each slot
        """
        self._init_projections()
        
        query = np.dot(query_hidden, self._query_projection)
        
        scores = []
        for slot in slots:
            semantic_score = np.dot(query, slot.key)
            
            if visual_context is not None:
                visual_score = float(np.dot(visual_context, self._visual_projection))
            else:
                visual_score = 0.0
            
            features = np.array([
                slot.retrieval_score,
                slot.reliability_score,
                slot.source_triple.confidence,
                1.0,
            ])
            feature_score = float(np.dot(features.flatten(), self._feature_projection.flatten()))
            
            reliability_logit = np.log(
                np.clip(slot.reliability_score, self.config.reliability_clamp_epsilon, 
                       1 - self.config.reliability_clamp_epsilon)
                / (1 - np.clip(slot.reliability_score, self.config.reliability_clamp_epsilon,
                              1 - self.config.reliability_clamp_epsilon))
            )
            reliability_bias = self.config.reliability_weight * reliability_logit + self.config.reliability_bias
            
            total_score = semantic_score + visual_score + feature_score + reliability_bias
            scores.append(total_score)
        
        return np.array(scores)
    
    def compute_gate_probabilities(
        self,
        utility_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gate probabilities from utility scores.
        
        u_p(t) = sigmoid(s_p(t))
        
        Args:
            utility_scores: Array of utility scores
            
        Returns:
            Array of gate probabilities in [0, 1]
        """
        return 1.0 / (1.0 + np.exp(-utility_scores))
    
    def select_active_slots(
        self,
        slots: List[HieraSlot],
        utility_scores: np.ndarray,
        gate_probabilities: np.ndarray,
    ) -> Tuple[List[HieraSlot], List[int]]:
        """
        Select active slots based on top-k utility and gate threshold.
        
        I_t = TopK({s_p(t)}, k)
        A_t = {p in I_t | u_p(t) > epsilon}
        
        Args:
            slots: List of all slots
            utility_scores: Utility scores for each slot
            gate_probabilities: Gate probabilities for each slot
            
        Returns:
            Tuple of (active slots, active indices)
        """
        top_k_indices = np.argsort(utility_scores)[-self.config.top_k_slots:][::-1]
        
        active_slots = []
        active_indices = []
        
        for idx in top_k_indices:
            if gate_probabilities[idx] > self.config.gate_threshold:
                active_slots.append(slots[idx])
                active_indices.append(int(idx))
        
        return active_slots, active_indices
    
    def route(
        self,
        slot_bank: SlotBank,
        query_hidden: np.ndarray,
        visual_context: Optional[np.ndarray] = None,
    ) -> RoutedSlots:
        """
        Route slots for a decoding step.
        
        Args:
            slot_bank: SlotBank containing all slots
            query_hidden: Query hidden state from decoder
            visual_context: Optional pooled visual context
            
        Returns:
            RoutedSlots containing active slots and routing info
        """
        all_slots = slot_bank.all_slots
        
        if not all_slots:
            return RoutedSlots(
                active_slots=[],
                gate_probabilities=np.array([]),
                utility_scores=np.array([]),
                active_indices=[],
            )
        
        utility_scores = self.compute_utility_scores(
            all_slots, query_hidden, visual_context
        )
        
        gate_probabilities = self.compute_gate_probabilities(utility_scores)
        
        active_slots, active_indices = self.select_active_slots(
            all_slots, utility_scores, gate_probabilities
        )
        
        return RoutedSlots(
            active_slots=active_slots,
            gate_probabilities=gate_probabilities,
            utility_scores=utility_scores,
            active_indices=active_indices,
        )


def reliability_calibrated_router(
    slots: SlotBank,
    question: str,
    router: Optional[ReliabilityCalibratedRouter] = None,
    query_hidden: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> RoutedSlots:
    """
    Convenience function for slot routing.
    
    Matches the pseudocode:
    routed = reliability_calibrated_router(slots, Q)
    
    Args:
        slots: SlotBank containing all slots
        question: Question text (used to generate query if not provided)
        router: Optional pre-configured router
        query_hidden: Optional query hidden state
        **kwargs: Arguments for router creation
        
    Returns:
        RoutedSlots containing active slots
    """
    if router is None:
        router = ReliabilityCalibratedRouter(**kwargs)
    
    if query_hidden is None:
        np.random.seed(hash(question) % (2**32))
        query_hidden = np.random.randn(router.hidden_dim)
        query_hidden = query_hidden / np.linalg.norm(query_hidden)
    
    return router.route(slots, query_hidden)


def select_top_k_slots(
    routed: RoutedSlots,
    k: Optional[int] = None,
) -> List[HieraSlot]:
    """
    Select top-k slots from routed result.
    
    Matches the pseudocode:
    top_slots = select_top_k_slots(routed, k = K_SLOTS)
    
    Args:
        routed: RoutedSlots from routing
        k: Number of slots to select (default: all active)
        
    Returns:
        List of top-k active slots
    """
    if k is None:
        return routed.active_slots
    return routed.active_slots[:k]
