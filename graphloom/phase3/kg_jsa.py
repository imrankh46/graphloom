"""
KG-JSA++ (Joint Graph-Sequence Attention) for GraphLoom Phase 3.

This module provides the joint attention mechanism that integrates external
evidence memories with decoder self-attention in a single joint operation.
"""

from typing import List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from graphloom.core.data_structures import HieraSlot


@dataclass
class JointAttentionOutput:
    """Output from KG-JSA++ joint attention."""
    output: np.ndarray
    attention_weights: np.ndarray
    prefix_attention: np.ndarray
    slot_attention: np.ndarray
    self_attention: np.ndarray


@dataclass
class PrefixMemory:
    """Multimodal prefix memory (fused question + facts + visual)."""
    keys: np.ndarray
    values: np.ndarray
    length: int


class KGJSAPlusPlus:
    """
    KG-JSA++: Joint Graph-Sequence Attention with Sparse Self-KV.
    
    Integrates external memories with decoder self-attention in a single
    joint operation within a frozen decoder. Key features:
    
    1. Memory Concatenation: Combines fused prefix KV with routed slot KV
    2. Sparse Self-KV Cache: Retains only recent, sink, and high-similarity tokens
    3. Joint Softmax: Single softmax over concatenated scores ensures competition
       between external evidence and model's own context
    
    This design provides:
    - O(L_p + |A_t| + |I_t|) complexity per step vs O(t) for full self-attention
    - ~76% reduction in attention FLOPs at t=512
    - Full access to critical evidence and recent context
    
    Example:
        >>> kg_jsa = KGJSAPlusPlus()
        >>> output = kg_jsa.forward(query, prefix_memory, routed_slots, kv_cache)
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128,
        prefix_length: int = 32,
        recent_tokens: int = 64,
        sink_tokens: int = 4,
        top_similarity_tokens: int = 20,
        reliability_gamma: float = 0.5,
        **kwargs: Any,
    ):
        """
        Initialize KG-JSA++.
        
        Args:
            hidden_dim: Hidden dimension of decoder
            num_heads: Number of attention heads
            head_dim: Dimension per attention head
            prefix_length: Fixed prefix memory length
            recent_tokens: Number of recent tokens to keep in cache
            sink_tokens: Number of sink tokens (first tokens) to keep
            top_similarity_tokens: Number of high-similarity tokens to keep
            reliability_gamma: Weight for reliability bias in attention
        """
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.prefix_length = prefix_length
        self.recent_tokens = recent_tokens
        self.sink_tokens = sink_tokens
        self.top_similarity_tokens = top_similarity_tokens
        self.reliability_gamma = reliability_gamma
        
        self._scale = 1.0 / np.sqrt(head_dim)
    
    def create_prefix_memory(
        self,
        question: str,
        facts_text: str,
        visual_features: Optional[np.ndarray] = None,
    ) -> PrefixMemory:
        """
        Create multimodal prefix memory from question, facts, and visual features.
        
        In the full implementation, this would:
        1. Encode question + facts with text encoder
        2. Fuse with visual tokens via cross-attention
        3. Project to fixed prefix length via learned pooling
        4. Convert to prefix key/value pairs
        
        Args:
            question: Question text
            facts_text: Concatenated facts from evidence subgraph
            visual_features: Optional visual features from image
            
        Returns:
            PrefixMemory containing keys and values
        """
        np.random.seed(hash(question + facts_text) % (2**32))
        
        keys = np.random.randn(self.prefix_length, self.hidden_dim) * 0.02
        values = np.random.randn(self.prefix_length, self.hidden_dim) * 0.02
        
        return PrefixMemory(
            keys=keys,
            values=values,
            length=self.prefix_length,
        )
    
    def concatenate_memories(
        self,
        prefix_memory: PrefixMemory,
        routed_slots: List[HieraSlot],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Concatenate prefix memory with routed slot KV pairs.
        
        K_pref(t) = [K_fus; K_t~]
        V_pref(t) = [V_fus; V_t~]
        
        Args:
            prefix_memory: Fused prefix memory
            routed_slots: List of active routed slots
            
        Returns:
            Tuple of (concatenated keys, concatenated values, reliability scores)
        """
        slot_keys = []
        slot_values = []
        reliability_scores = []
        
        for slot in routed_slots:
            slot_keys.append(slot.key)
            slot_values.append(slot.value)
            reliability_scores.append(slot.reliability_score)
        
        if slot_keys:
            slot_keys = np.stack(slot_keys)
            slot_values = np.stack(slot_values)
            reliability_scores = np.array(reliability_scores)
            
            concat_keys = np.concatenate([prefix_memory.keys, slot_keys], axis=0)
            concat_values = np.concatenate([prefix_memory.values, slot_values], axis=0)
        else:
            concat_keys = prefix_memory.keys
            concat_values = prefix_memory.values
            reliability_scores = np.array([])
        
        return concat_keys, concat_values, reliability_scores
    
    def sparsify_self_kv_cache(
        self,
        query: np.ndarray,
        kv_cache_keys: np.ndarray,
        kv_cache_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sparsify self-attention KV cache.
        
        Retains only:
        - Sink tokens: First 4 tokens (attention sinks)
        - Recent tokens: Last 64 tokens
        - High-similarity tokens: Top-20 by cosine similarity to query
        
        Args:
            query: Current query vector
            kv_cache_keys: Full KV cache keys
            kv_cache_values: Full KV cache values
            
        Returns:
            Tuple of (sparse keys, sparse values, selected indices)
        """
        seq_len = kv_cache_keys.shape[0]
        
        if seq_len <= (self.sink_tokens + self.recent_tokens + self.top_similarity_tokens):
            indices = np.arange(seq_len)
            return kv_cache_keys, kv_cache_values, indices
        
        sink_indices = set(range(min(self.sink_tokens, seq_len)))
        
        recent_start = max(0, seq_len - self.recent_tokens)
        recent_indices = set(range(recent_start, seq_len))
        
        middle_start = self.sink_tokens
        middle_end = recent_start
        
        if middle_end > middle_start:
            middle_keys = kv_cache_keys[middle_start:middle_end]
            similarities = np.dot(middle_keys, query) / (
                np.linalg.norm(middle_keys, axis=1) * np.linalg.norm(query) + 1e-9
            )
            top_k = min(self.top_similarity_tokens, len(similarities))
            top_middle_indices = np.argsort(similarities)[-top_k:]
            similarity_indices = set(top_middle_indices + middle_start)
        else:
            similarity_indices = set()
        
        all_indices = sorted(sink_indices | recent_indices | similarity_indices)
        indices = np.array(all_indices)
        
        sparse_keys = kv_cache_keys[indices]
        sparse_values = kv_cache_values[indices]
        
        return sparse_keys, sparse_values, indices
    
    def compute_attention_scores(
        self,
        query: np.ndarray,
        prefix_keys: np.ndarray,
        self_keys: np.ndarray,
        reliability_scores: np.ndarray,
        causal_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attention scores for prefix and self-attention.
        
        S_pref = (q * K_pref^T) / sqrt(d) + B_pref(t)
        S_self = (q * K_self^T) / sqrt(d) + M_causal
        
        Args:
            query: Query vector
            prefix_keys: Concatenated prefix + slot keys
            self_keys: Sparse self-attention keys
            reliability_scores: Reliability scores for slots
            causal_mask: Optional causal attention mask
            
        Returns:
            Tuple of (prefix scores, self scores)
        """
        prefix_scores = np.dot(query, prefix_keys.T) * self._scale
        
        if len(reliability_scores) > 0:
            prefix_len = prefix_keys.shape[0] - len(reliability_scores)
            reliability_bias = np.zeros(prefix_keys.shape[0])
            
            for i, rel_score in enumerate(reliability_scores):
                clamped = np.clip(rel_score, 0.1, 0.9)
                logit = np.log(clamped / (1 - clamped))
                reliability_bias[prefix_len + i] = self.reliability_gamma * logit
            
            prefix_scores = prefix_scores + reliability_bias
        
        self_scores = np.dot(query, self_keys.T) * self._scale
        
        if causal_mask is not None:
            self_scores = self_scores + causal_mask
        
        return prefix_scores, self_scores
    
    def joint_softmax(
        self,
        prefix_scores: np.ndarray,
        self_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Compute joint softmax over concatenated scores.
        
        A_t = softmax([S_pref; S_self])
        
        This ensures competition between external evidence and model's own context.
        
        Args:
            prefix_scores: Attention scores for prefix + slots
            self_scores: Attention scores for self-attention
            
        Returns:
            Joint attention weights
        """
        all_scores = np.concatenate([prefix_scores, self_scores])
        
        max_score = np.max(all_scores)
        exp_scores = np.exp(all_scores - max_score)
        attention_weights = exp_scores / (np.sum(exp_scores) + 1e-9)
        
        return attention_weights
    
    def forward(
        self,
        query: np.ndarray,
        prefix_memory: PrefixMemory,
        routed_slots: List[HieraSlot],
        kv_cache_keys: Optional[np.ndarray] = None,
        kv_cache_values: Optional[np.ndarray] = None,
    ) -> JointAttentionOutput:
        """
        Forward pass of KG-JSA++ joint attention.
        
        Args:
            query: Query vector from decoder hidden state
            prefix_memory: Fused multimodal prefix memory
            routed_slots: List of active routed slots
            kv_cache_keys: Optional self-attention KV cache keys
            kv_cache_values: Optional self-attention KV cache values
            
        Returns:
            JointAttentionOutput containing output and attention info
        """
        prefix_keys, prefix_values, reliability_scores = self.concatenate_memories(
            prefix_memory, routed_slots
        )
        
        if kv_cache_keys is not None and kv_cache_values is not None:
            self_keys, self_values, _ = self.sparsify_self_kv_cache(
                query, kv_cache_keys, kv_cache_values
            )
        else:
            self_keys = np.zeros((0, self.hidden_dim))
            self_values = np.zeros((0, self.hidden_dim))
        
        prefix_scores, self_scores = self.compute_attention_scores(
            query, prefix_keys, self_keys, reliability_scores
        )
        
        attention_weights = self.joint_softmax(prefix_scores, self_scores)
        
        all_values = np.concatenate([prefix_values, self_values], axis=0)
        output = np.dot(attention_weights, all_values)
        
        prefix_len = prefix_keys.shape[0]
        prefix_attention = attention_weights[:prefix_len]
        self_attention = attention_weights[prefix_len:]
        
        slot_start = prefix_memory.length
        slot_attention = prefix_attention[slot_start:]
        
        return JointAttentionOutput(
            output=output,
            attention_weights=attention_weights,
            prefix_attention=prefix_attention,
            slot_attention=slot_attention,
            self_attention=self_attention,
        )


def kg_jsa_pp_attention(
    image: Any,
    question: str,
    slots: List[HieraSlot],
    evidence_text: str,
    kg_jsa: Optional[KGJSAPlusPlus] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Convenience function for KG-JSA++ attention.
    
    Matches the pseudocode:
    joint_context = KG_JSA_pp_attention(
        image = I,
        question = Q,
        slots = top_slots
    )
    
    Args:
        image: Input image (for visual features)
        question: Question text
        slots: List of routed slots
        evidence_text: Text representation of evidence
        kg_jsa: Optional pre-configured KG-JSA++ module
        **kwargs: Arguments for KG-JSA++ creation
        
    Returns:
        Joint context vector
    """
    if kg_jsa is None:
        kg_jsa = KGJSAPlusPlus(**kwargs)
    
    prefix_memory = kg_jsa.create_prefix_memory(question, evidence_text)
    
    np.random.seed(hash(question) % (2**32))
    query = np.random.randn(kg_jsa.hidden_dim)
    query = query / np.linalg.norm(query)
    
    output = kg_jsa.forward(query, prefix_memory, slots)
    
    return output.output
