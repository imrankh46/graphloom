"""
Triple Extractor for GraphLoom Phase 1.

This module provides triple extraction from text using REBEL (Relation Extraction By End-to-end Language generation).
"""

from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import re

from graphloom.core.data_structures import Triple


class BaseTripleExtractor(ABC):
    """Abstract base class for triple extractors."""
    
    @abstractmethod
    def extract(self, text: str, **kwargs: Any) -> List[Triple]:
        """
        Extract triples from text.
        
        Args:
            text: Input text to extract triples from
            **kwargs: Additional arguments
            
        Returns:
            List of extracted Triple objects
        """
        pass


class REBELExtractor(BaseTripleExtractor):
    """
    Triple extractor using REBEL model.
    
    REBEL is a transformer-based relation extractor that translates
    natural language into (subject, relation, object) knowledge triples.
    """
    
    def __init__(
        self,
        model_name: str = "Babelscape/rebel-large",
        device: str = "auto",
        confidence_threshold: float = 0.5,
        max_length: int = 512,
    ):
        """
        Initialize the REBEL extractor.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on
            confidence_threshold: Minimum confidence for triple inclusion
            max_length: Maximum input sequence length
        """
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
    
    def _load_model(self) -> None:
        """Lazy load the model and tokenizer."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(self.device)
            self._model.eval()
        except ImportError as e:
            raise ImportError(
                "Please install transformers and torch: "
                "pip install transformers torch"
            ) from e
    
    def _parse_rebel_output(self, text: str) -> List[Tuple[str, str, str, float]]:
        """
        Parse REBEL model output into triples.
        
        REBEL outputs triples in a specific format:
        <triplet> subject <subj> relation <obj> object
        
        Args:
            text: Raw model output text
            
        Returns:
            List of (subject, relation, object, confidence) tuples
        """
        triples = []
        
        triplet_pattern = r'<triplet>\s*([^<]+?)\s*<subj>\s*([^<]+?)\s*<obj>\s*([^<]+?)(?=<triplet>|$)'
        matches = re.findall(triplet_pattern, text, re.DOTALL)
        
        for match in matches:
            if len(match) == 3:
                subject = match[0].strip()
                relation = match[1].strip()
                obj = match[2].strip()
                
                if subject and relation and obj:
                    confidence = 0.8
                    triples.append((subject, relation, obj, confidence))
        
        return triples
    
    def extract(self, text: str, **kwargs: Any) -> List[Triple]:
        """
        Extract triples from text using REBEL.
        
        Args:
            text: Input text to extract triples from
            **kwargs: Additional generation arguments
            
        Returns:
            List of extracted Triple objects
        """
        self._load_model()
        
        import torch
        
        inputs = self._tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=kwargs.get("max_length", 256),
                num_beams=kwargs.get("num_beams", 5),
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        decoded = self._tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=False,
        )[0]
        
        raw_triples = self._parse_rebel_output(decoded)
        
        triples = []
        for subject, relation, obj, confidence in raw_triples:
            if confidence >= self.confidence_threshold:
                triple = Triple(
                    subject=subject,
                    relation=relation,
                    object=obj,
                    confidence=confidence,
                    source="rebel",
                )
                triples.append(triple)
        
        return triples


class SpacyTripleExtractor(BaseTripleExtractor):
    """
    Simple triple extractor using spaCy dependency parsing.
    
    Extracts subject-verb-object triples from sentences using
    dependency parse trees. Less accurate than REBEL but faster
    and doesn't require a large model.
    """
    
    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the spaCy extractor.
        
        Args:
            model_name: spaCy model name
            confidence_threshold: Minimum confidence for triple inclusion
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._nlp = None
    
    def _load_model(self) -> None:
        """Lazy load the spaCy model."""
        if self._nlp is not None:
            return
        
        try:
            import spacy
            self._nlp = spacy.load(self.model_name)
        except ImportError:
            raise ImportError("Please install spacy: pip install spacy")
        except OSError:
            raise OSError(
                f"spaCy model '{self.model_name}' not found. "
                f"Run: python -m spacy download {self.model_name}"
            )
    
    def extract(self, text: str, **kwargs: Any) -> List[Triple]:
        """
        Extract triples from text using spaCy dependency parsing.
        
        Args:
            text: Input text to extract triples from
            **kwargs: Additional arguments (unused)
            
        Returns:
            List of extracted Triple objects
        """
        self._load_model()
        
        doc = self._nlp(text)
        triples = []
        
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB":
                    subjects = []
                    objects = []
                    
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            subject_span = self._get_span(child)
                            subjects.append(subject_span)
                        elif child.dep_ in ("dobj", "pobj", "attr"):
                            object_span = self._get_span(child)
                            objects.append(object_span)
                    
                    for subj in subjects:
                        for obj in objects:
                            if subj and obj:
                                triple = Triple(
                                    subject=subj,
                                    relation=token.lemma_,
                                    object=obj,
                                    confidence=0.6,
                                    source="spacy",
                                )
                                triples.append(triple)
        
        return triples
    
    def _get_span(self, token) -> str:
        """Get the full noun phrase span for a token."""
        subtree = list(token.subtree)
        start = subtree[0].i
        end = subtree[-1].i + 1
        return token.doc[start:end].text


class MockTripleExtractor(BaseTripleExtractor):
    """
    Mock extractor for testing without model dependencies.
    
    Returns predefined triples based on simple pattern matching.
    """
    
    def __init__(self, **kwargs):
        pass
    
    def extract(self, text: str, **kwargs: Any) -> List[Triple]:
        """Return mock triples for testing."""
        words = text.lower().split()
        triples = []
        
        if len(words) >= 3:
            triple = Triple(
                subject=words[0].capitalize(),
                relation="relates_to",
                object=words[-1].rstrip("."),
                confidence=0.9,
                source="mock",
            )
            triples.append(triple)
        
        return triples


def create_triple_extractor(
    extractor_type: str = "rebel",
    **kwargs: Any,
) -> BaseTripleExtractor:
    """
    Factory function to create a triple extractor.
    
    Args:
        extractor_type: Type of extractor ('rebel', 'spacy', 'mock')
        **kwargs: Arguments passed to extractor constructor
        
    Returns:
        Configured triple extractor instance
    """
    extractors = {
        "rebel": REBELExtractor,
        "spacy": SpacyTripleExtractor,
        "mock": MockTripleExtractor,
    }
    
    if extractor_type not in extractors:
        raise ValueError(f"Unknown extractor type: {extractor_type}. "
                        f"Available: {list(extractors.keys())}")
    
    return extractors[extractor_type](**kwargs)


def clean_factual_description(text: str) -> str:
    """
    Clean and normalize a factual description for triple extraction.
    
    Args:
        text: Raw description text
        
    Returns:
        Cleaned text suitable for triple extraction
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s.,;:!?\'"()-]', '', text)
    
    return text
