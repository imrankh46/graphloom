"""
Phase 1 Pipeline: Unified MMKG Build.

This module orchestrates the Phase 1 pipeline:
1. Multimodal understanding with Qwen3-VL-Instruct
2. Clean factual description generation
3. Triple extraction with REBEL
4. Direct insertion into MMKG (no enrichment, no canonicalization)
"""

from typing import Union, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np

from graphloom.core.data_structures import Triple, MMKG
from graphloom.phase1.multimodal_encoder import (
    BaseMultimodalEncoder,
    create_multimodal_encoder,
)
from graphloom.phase1.triple_extractor import (
    BaseTripleExtractor,
    create_triple_extractor,
    clean_factual_description,
)


class Phase1Pipeline:
    """
    Phase 1: Unified MMKG Build Pipeline.
    
    This pipeline handles:
    1. Multimodal understanding - Generate unified scene description
    2. Factual description cleaning
    3. Triple extraction from description
    4. Direct MMKG insertion (simplified, no enrichment)
    
    Example:
        >>> pipeline = Phase1Pipeline()
        >>> mmkg = MMKG()
        >>> mmkg, triples = pipeline.process(image, question, mmkg)
    """
    
    def __init__(
        self,
        multimodal_encoder: Optional[BaseMultimodalEncoder] = None,
        triple_extractor: Optional[BaseTripleExtractor] = None,
        encoder_type: str = "mock",
        extractor_type: str = "mock",
        confidence_threshold: float = 0.5,
        **kwargs: Any,
    ):
        """
        Initialize the Phase 1 pipeline.
        
        Args:
            multimodal_encoder: Pre-configured multimodal encoder (optional)
            triple_extractor: Pre-configured triple extractor (optional)
            encoder_type: Type of encoder to create if not provided
            extractor_type: Type of extractor to create if not provided
            confidence_threshold: Minimum confidence for triple inclusion
            **kwargs: Additional arguments for encoder/extractor creation
        """
        self.multimodal_encoder = multimodal_encoder or create_multimodal_encoder(
            encoder_type=encoder_type,
            **kwargs.get("encoder_kwargs", {}),
        )
        
        self.triple_extractor = triple_extractor or create_triple_extractor(
            extractor_type=extractor_type,
            confidence_threshold=confidence_threshold,
            **kwargs.get("extractor_kwargs", {}),
        )
        
        self.confidence_threshold = confidence_threshold
    
    def generate_description(
        self,
        image: Union[str, Path, np.ndarray],
        question: str,
    ) -> str:
        """
        Generate a unified factual description from image and question.
        
        Args:
            image: Image path, URL, or numpy array
            question: The question about the image
            
        Returns:
            Unified factual description text
        """
        raw_description = self.multimodal_encoder.encode(image, question)
        clean_description = clean_factual_description(raw_description)
        return clean_description
    
    def extract_triples(self, description: str) -> List[Triple]:
        """
        Extract triples from a factual description.
        
        Args:
            description: Factual description text
            
        Returns:
            List of extracted Triple objects
        """
        triples = self.triple_extractor.extract(description)
        filtered_triples = [
            t for t in triples
            if t.confidence >= self.confidence_threshold
        ]
        return filtered_triples
    
    def insert_triples(
        self,
        mmkg: MMKG,
        triples: List[Triple],
    ) -> int:
        """
        Insert extracted triples directly into MMKG.
        
        No enrichment or canonicalization is performed in this simplified version.
        
        Args:
            mmkg: The MMKG to insert triples into
            triples: List of triples to insert
            
        Returns:
            Number of triples successfully added
        """
        return mmkg.add_batch(triples)
    
    def process(
        self,
        image: Union[str, Path, np.ndarray],
        question: str,
        mmkg: MMKG,
        return_description: bool = False,
    ) -> Union[Tuple[MMKG, List[Triple]], Tuple[MMKG, List[Triple], str]]:
        """
        Execute the full Phase 1 pipeline.
        
        Steps:
        1. Generate unified factual description from image and question
        2. Clean the description
        3. Extract triples using REBEL
        4. Insert triples directly into MMKG
        
        Args:
            image: Image path, URL, or numpy array
            question: The question about the image
            mmkg: The MMKG to update with extracted triples
            return_description: Whether to return the generated description
            
        Returns:
            Tuple of (updated MMKG, extracted triples) or
            Tuple of (updated MMKG, extracted triples, description) if return_description=True
        """
        description = self.generate_description(image, question)
        
        triples = self.extract_triples(description)
        
        for triple in triples:
            triple.metadata["question"] = question
            if isinstance(image, (str, Path)):
                triple.metadata["image_path"] = str(image)
        
        self.insert_triples(mmkg, triples)
        
        if return_description:
            return mmkg, triples, description
        return mmkg, triples


def phase_1_unified(
    image: Union[str, Path, np.ndarray],
    question: str,
    mmkg: MMKG,
    pipeline: Optional[Phase1Pipeline] = None,
    **kwargs: Any,
) -> Tuple[MMKG, List[Triple]]:
    """
    Convenience function for Phase 1 processing.
    
    Matches the pseudocode signature:
    function PHASE_1_UNIFIED(image I, question Q, MMKG):
        ...
        return MMKG, extracted_triples
    
    Args:
        image: Image path, URL, or numpy array
        question: The question about the image
        mmkg: The MMKG to update
        pipeline: Optional pre-configured pipeline
        **kwargs: Arguments for pipeline creation
        
    Returns:
        Tuple of (updated MMKG, extracted triples)
    """
    if pipeline is None:
        pipeline = Phase1Pipeline(**kwargs)
    
    return pipeline.process(image, question, mmkg)
