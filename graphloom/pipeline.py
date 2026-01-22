"""
GraphLoom Main Pipeline.

This module provides the unified GraphLoom pipeline that orchestrates
all three phases for multimodal question answering.
"""

from typing import Union, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np

from graphloom.core.data_structures import (
    Triple,
    MMKG,
    EvidenceSubgraph,
    RetrievalConfig,
    RouterConfig,
)
from graphloom.phase1.phase1_pipeline import Phase1Pipeline
from graphloom.phase2.phase2_pipeline import Phase2Pipeline
from graphloom.phase3.phase3_pipeline import Phase3Pipeline
from graphloom.utils.config import GraphLoomConfig


class GraphLoomPipeline:
    """
    GraphLoom: Unified Multimodal KG-RAG Pipeline.
    
    A multimodal QA framework that unifies retrieval-augmented generation,
    instance-level multimodal knowledge graphs, and frozen large language models.
    
    Pipeline Overview:
    1. Phase 1 (MMKG Build): Generate description + extract triples + insert into MMKG
    2. Phase 2 (Retrieval): Embed query + score triples + bounded expansion
    3. Phase 3 (Reasoning): Encode slots + route + KG-JSA++ attention + generate answer
    
    Example:
        >>> pipeline = GraphLoomPipeline()
        >>> mmkg = MMKG()
        >>> answer = pipeline.answer_question(image, question, mmkg)
        
        # Or with config
        >>> config = GraphLoomConfig.from_preset("production")
        >>> pipeline = GraphLoomPipeline(config=config)
    """
    
    def __init__(
        self,
        config: Optional[GraphLoomConfig] = None,
        phase1: Optional[Phase1Pipeline] = None,
        phase2: Optional[Phase2Pipeline] = None,
        phase3: Optional[Phase3Pipeline] = None,
        **kwargs: Any,
    ):
        """
        Initialize the GraphLoom pipeline.
        
        Args:
            config: GraphLoom configuration (optional, uses mock preset if None)
            phase1: Pre-configured Phase 1 pipeline (optional)
            phase2: Pre-configured Phase 2 pipeline (optional)
            phase3: Pre-configured Phase 3 pipeline (optional)
            **kwargs: Additional arguments for pipeline creation
        """
        self.config = config or GraphLoomConfig.from_preset("mock")
        
        self.phase1 = phase1 or self._create_phase1()
        self.phase2 = phase2 or self._create_phase2()
        self.phase3 = phase3 or self._create_phase3()
    
    def _create_phase1(self) -> Phase1Pipeline:
        """Create Phase 1 pipeline from config."""
        return Phase1Pipeline(
            encoder_type=self.config.phase1.encoder_type,
            extractor_type=self.config.phase1.extractor_type,
            confidence_threshold=self.config.phase1.confidence_threshold,
            encoder_kwargs=self.config.phase1.encoder_kwargs,
            extractor_kwargs=self.config.phase1.extractor_kwargs,
        )
    
    def _create_phase2(self) -> Phase2Pipeline:
        """Create Phase 2 pipeline from config."""
        retrieval_config = RetrievalConfig(
            top_k=self.config.phase2.top_k,
            max_hops=self.config.phase2.max_hops,
            max_degree=self.config.phase2.max_degree,
            max_edges=self.config.phase2.max_edges,
            similarity_threshold=self.config.phase2.similarity_threshold,
        )
        return Phase2Pipeline(
            embedder_type=self.config.phase2.embedder_type,
            config=retrieval_config,
            embedder_kwargs=self.config.phase2.embedder_kwargs,
        )
    
    def _create_phase3(self) -> Phase3Pipeline:
        """Create Phase 3 pipeline from config."""
        router_config = RouterConfig(
            top_k_slots=self.config.phase3.top_k_slots,
            gate_threshold=self.config.phase3.gate_threshold,
            reliability_weight=self.config.phase3.reliability_weight,
        )
        return Phase3Pipeline(
            generator_type=self.config.phase3.generator_type,
            hidden_dim=self.config.phase3.hidden_dim,
            router_config=router_config,
            generator_kwargs=self.config.phase3.generator_kwargs,
        )
    
    def run_phase1(
        self,
        image: Union[str, Path, np.ndarray],
        question: str,
        mmkg: MMKG,
    ) -> Tuple[MMKG, List[Triple]]:
        """
        Run Phase 1: Unified MMKG Build.
        
        Steps:
        1. Multimodal understanding with Qwen3-VL-Instruct
        2. Clean factual description
        3. Triple extraction with REBEL
        4. Insert triples directly into MMKG
        
        Args:
            image: Input image
            question: Question text
            mmkg: MMKG to update
            
        Returns:
            Tuple of (updated MMKG, extracted triples)
        """
        return self.phase1.process(image, question, mmkg)
    
    def run_phase2(
        self,
        mmkg: MMKG,
        extracted_triples: List[Triple],
        image: Union[str, Path, np.ndarray],
        question: str,
    ) -> EvidenceSubgraph:
        """
        Run Phase 2: Evidence Subgraph Retrieval.
        
        Steps:
        1. Compute multimodal query embedding
        2. Retrieve and score candidate triples
        3. Select top-k seed triples
        4. Perform bounded graph expansion
        
        Args:
            mmkg: MMKG to retrieve from
            extracted_triples: Triples from Phase 1
            image: Input image
            question: Question text
            
        Returns:
            EvidenceSubgraph containing compact evidence
        """
        return self.phase2.process(mmkg, extracted_triples, image, question)
    
    def run_phase3(
        self,
        evidence_subgraph: EvidenceSubgraph,
        image: Union[str, Path, np.ndarray],
        question: str,
    ) -> str:
        """
        Run Phase 3: Controlled Reasoning.
        
        Steps:
        1. Encode evidence into HieraSlot memories
        2. Route slots with reliability calibration
        3. Compute joint context with KG-JSA++
        4. Generate faithful answer
        
        Args:
            evidence_subgraph: Evidence from Phase 2
            image: Input image
            question: Question text
            
        Returns:
            Generated answer string
        """
        return self.phase3.process(evidence_subgraph, image, question)
    
    def answer_question(
        self,
        image: Union[str, Path, np.ndarray],
        question: str,
        mmkg: MMKG,
        return_evidence: bool = False,
    ) -> Union[str, Tuple[str, EvidenceSubgraph]]:
        """
        Answer a question using the full GraphLoom pipeline.
        
        This is the main entry point matching the pseudocode:
        
        function ANSWER_QUESTION(image I, question Q, MMKG):
            MMKG, triples = PHASE_1_UNIFIED(I, Q, MMKG)
            subgraph = PHASE_2_SIMPLIFIED(MMKG, triples, I, Q)
            answer = PHASE_3_CONTROLLED(subgraph, I, Q)
            return answer
        
        Args:
            image: Input image (path, URL, or numpy array)
            question: Question text
            mmkg: MMKG to use (will be updated with extracted triples)
            return_evidence: Whether to return evidence subgraph
            
        Returns:
            Answer string, or tuple of (answer, evidence_subgraph) if return_evidence=True
        """
        mmkg, extracted_triples = self.run_phase1(image, question, mmkg)
        
        evidence_subgraph = self.run_phase2(mmkg, extracted_triples, image, question)
        
        answer = self.run_phase3(evidence_subgraph, image, question)
        
        if return_evidence:
            return answer, evidence_subgraph
        return answer
    
    def batch_answer(
        self,
        qa_pairs: List[Tuple[Any, str]],
        mmkg: MMKG,
    ) -> List[str]:
        """
        Answer multiple questions in batch.
        
        Args:
            qa_pairs: List of (image, question) tuples
            mmkg: Shared MMKG (will be updated with all extracted triples)
            
        Returns:
            List of answer strings
        """
        answers = []
        for image, question in qa_pairs:
            answer = self.answer_question(image, question, mmkg)
            answers.append(answer)
        return answers


def answer_question(
    image: Union[str, Path, np.ndarray],
    question: str,
    mmkg: MMKG,
    pipeline: Optional[GraphLoomPipeline] = None,
    **kwargs: Any,
) -> str:
    """
    Convenience function for answering a question.
    
    Matches the pseudocode signature:
    function ANSWER_QUESTION(image I, question Q, MMKG):
        MMKG, triples = PHASE_1_UNIFIED(I, Q, MMKG)
        subgraph = PHASE_2_SIMPLIFIED(MMKG, triples, I, Q)
        answer = PHASE_3_CONTROLLED(subgraph, I, Q)
        return answer
    
    Args:
        image: Input image
        question: Question text
        mmkg: MMKG to use
        pipeline: Optional pre-configured pipeline
        **kwargs: Arguments for pipeline creation
        
    Returns:
        Generated answer string
    """
    if pipeline is None:
        config = kwargs.pop("config", None)
        pipeline = GraphLoomPipeline(config=config, **kwargs)
    
    return pipeline.answer_question(image, question, mmkg)
