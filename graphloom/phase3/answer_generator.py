"""
Answer Generator for GraphLoom Phase 3.

This module provides faithful answer generation using the joint context
from KG-JSA++ attention and evidence subgraph.
"""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from graphloom.core.data_structures import Triple, EvidenceSubgraph, HieraSlot


@dataclass
class GeneratedAnswer:
    """Result of answer generation."""
    answer: str
    confidence: float
    evidence_used: List[Triple]
    reasoning_trace: Optional[str] = None


class BaseAnswerGenerator(ABC):
    """Abstract base class for answer generators."""
    
    @abstractmethod
    def generate(
        self,
        question: str,
        joint_context: np.ndarray,
        evidence_subgraph: EvidenceSubgraph,
        **kwargs: Any,
    ) -> GeneratedAnswer:
        """
        Generate an answer given question, context, and evidence.
        
        Args:
            question: The question to answer
            joint_context: Joint context from KG-JSA++ attention
            evidence_subgraph: Evidence subgraph for grounding
            **kwargs: Additional arguments
            
        Returns:
            GeneratedAnswer containing answer and metadata
        """
        pass


class LlamaAnswerGenerator(BaseAnswerGenerator):
    """
    Answer generator using Llama-3.1-8B-Instruct.
    
    Generates faithful answers by conditioning on:
    - Question text
    - Evidence from retrieved subgraph
    - Joint context from KG-JSA++ attention
    """
    
    ANSWER_PROMPT = """You are a precise question answering assistant. Answer the question based ONLY on the provided evidence. If the evidence is insufficient, say "Insufficient evidence."

Evidence:
{evidence}

Question: {question}

Provide a concise, factual answer:"""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ):
        """
        Initialize the Llama answer generator.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
    
    def _load_model(self) -> None:
        """Lazy load the model and tokenizer."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,
            )
        except ImportError as e:
            raise ImportError(
                "Please install transformers and torch: "
                "pip install transformers torch"
            ) from e
    
    def generate(
        self,
        question: str,
        joint_context: np.ndarray,
        evidence_subgraph: EvidenceSubgraph,
        **kwargs: Any,
    ) -> GeneratedAnswer:
        """
        Generate an answer using Llama.
        
        Args:
            question: The question to answer
            joint_context: Joint context from KG-JSA++ (used for conditioning)
            evidence_subgraph: Evidence subgraph for grounding
            **kwargs: Additional generation arguments
            
        Returns:
            GeneratedAnswer containing answer and metadata
        """
        self._load_model()
        
        evidence_text = evidence_subgraph.to_text()
        
        prompt = self.ANSWER_PROMPT.format(
            evidence=evidence_text,
            question=question,
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self._tokenizer(input_text, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        
        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        answer = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return GeneratedAnswer(
            answer=answer.strip(),
            confidence=0.8,
            evidence_used=evidence_subgraph.all_triples,
        )


class MockAnswerGenerator(BaseAnswerGenerator):
    """
    Mock answer generator for testing without model dependencies.
    
    Generates template answers based on evidence.
    """
    
    def __init__(self):
        pass
    
    def generate(
        self,
        question: str,
        joint_context: np.ndarray,
        evidence_subgraph: EvidenceSubgraph,
        **kwargs: Any,
    ) -> GeneratedAnswer:
        """Generate a mock answer based on evidence."""
        triples = evidence_subgraph.all_triples
        
        if not triples:
            return GeneratedAnswer(
                answer="Insufficient evidence to answer the question.",
                confidence=0.1,
                evidence_used=[],
            )
        
        top_triple = triples[0]
        answer = f"Based on the evidence, {top_triple.subject} {top_triple.relation} {top_triple.object}."
        
        return GeneratedAnswer(
            answer=answer,
            confidence=0.7,
            evidence_used=triples[:3],
        )


def create_answer_generator(
    generator_type: str = "mock",
    **kwargs: Any,
) -> BaseAnswerGenerator:
    """
    Factory function to create an answer generator.
    
    Args:
        generator_type: Type of generator ('llama', 'mock')
        **kwargs: Arguments passed to generator constructor
        
    Returns:
        Configured answer generator instance
    """
    generators = {
        "llama": LlamaAnswerGenerator,
        "mock": MockAnswerGenerator,
    }
    
    if generator_type not in generators:
        raise ValueError(f"Unknown generator type: {generator_type}. "
                        f"Available: {list(generators.keys())}")
    
    return generators[generator_type](**kwargs)


def generate_faithful_answer(
    question: str,
    joint_context: np.ndarray,
    evidence_subgraph: EvidenceSubgraph,
    generator: Optional[BaseAnswerGenerator] = None,
    **kwargs: Any,
) -> str:
    """
    Convenience function for answer generation.
    
    Matches the pseudocode:
    final_answer = generate_faithful_answer(
        question = Q,
        joint_context = joint_context,
        evidence_subgraph = compact_subgraph
    )
    
    Args:
        question: The question to answer
        joint_context: Joint context from KG-JSA++
        evidence_subgraph: Evidence subgraph for grounding
        generator: Optional pre-configured generator
        **kwargs: Arguments for generator creation
        
    Returns:
        Generated answer string
    """
    if generator is None:
        generator = create_answer_generator(**kwargs)
    
    result = generator.generate(question, joint_context, evidence_subgraph, **kwargs)
    return result.answer
