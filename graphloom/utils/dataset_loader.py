"""
Dataset Loader for GraphLoom.

This module provides utilities for loading datasets and building MMKGs
from existing data without enrichment.
"""

from typing import List, Dict, Any, Optional, Iterator, Tuple
from pathlib import Path
import json

from graphloom.core.data_structures import Triple, MMKG


class DatasetLoader:
    """
    Dataset loader for building MMKGs from existing datasets.
    
    Supports loading from:
    - JSON files with triple annotations
    - CSV files with subject, relation, object columns
    - Custom formats via subclassing
    
    Example:
        >>> loader = DatasetLoader()
        >>> mmkg = loader.load_from_json("triples.json")
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.0,
        default_confidence: float = 1.0,
        default_source: str = "dataset",
    ):
        """
        Initialize the dataset loader.
        
        Args:
            confidence_threshold: Minimum confidence for triple inclusion
            default_confidence: Default confidence for triples without scores
            default_source: Default source label for loaded triples
        """
        self.confidence_threshold = confidence_threshold
        self.default_confidence = default_confidence
        self.default_source = default_source
    
    def parse_triple(
        self,
        data: Dict[str, Any],
        source: Optional[str] = None,
    ) -> Optional[Triple]:
        """
        Parse a triple from a dictionary.
        
        Expected keys: subject, relation/predicate, object
        Optional keys: confidence, source, metadata
        
        Args:
            data: Dictionary containing triple data
            source: Optional source override
            
        Returns:
            Triple object or None if invalid
        """
        subject = data.get("subject") or data.get("head") or data.get("s")
        relation = data.get("relation") or data.get("predicate") or data.get("r") or data.get("p")
        obj = data.get("object") or data.get("tail") or data.get("o")
        
        if not all([subject, relation, obj]):
            return None
        
        confidence = data.get("confidence", data.get("score", self.default_confidence))
        
        if confidence < self.confidence_threshold:
            return None
        
        return Triple(
            subject=str(subject).strip(),
            relation=str(relation).strip(),
            object=str(obj).strip(),
            confidence=float(confidence),
            source=source or data.get("source", self.default_source),
            metadata=data.get("metadata", {}),
        )
    
    def load_from_json(
        self,
        path: str,
        triples_key: Optional[str] = None,
    ) -> MMKG:
        """
        Load MMKG from a JSON file.
        
        Supports formats:
        - List of triples: [{"subject": ..., "relation": ..., "object": ...}, ...]
        - Object with triples key: {"triples": [...], ...}
        
        Args:
            path: Path to JSON file
            triples_key: Key containing triples list (auto-detected if None)
            
        Returns:
            MMKG populated with loaded triples
        """
        with open(path, "r") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            triples_data = data
        elif triples_key:
            triples_data = data[triples_key]
        else:
            for key in ["triples", "data", "facts", "edges"]:
                if key in data and isinstance(data[key], list):
                    triples_data = data[key]
                    break
            else:
                raise ValueError(
                    f"Could not find triples in JSON. "
                    f"Specify triples_key or use a list format."
                )
        
        mmkg = MMKG(confidence_threshold=self.confidence_threshold)
        
        for item in triples_data:
            triple = self.parse_triple(item)
            if triple:
                mmkg.add(triple)
        
        return mmkg
    
    def load_from_csv(
        self,
        path: str,
        subject_col: str = "subject",
        relation_col: str = "relation",
        object_col: str = "object",
        confidence_col: Optional[str] = None,
        delimiter: str = ",",
    ) -> MMKG:
        """
        Load MMKG from a CSV file.
        
        Args:
            path: Path to CSV file
            subject_col: Column name for subjects
            relation_col: Column name for relations
            object_col: Column name for objects
            confidence_col: Optional column name for confidence scores
            delimiter: CSV delimiter
            
        Returns:
            MMKG populated with loaded triples
        """
        import csv
        
        mmkg = MMKG(confidence_threshold=self.confidence_threshold)
        
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row in reader:
                data = {
                    "subject": row.get(subject_col),
                    "relation": row.get(relation_col),
                    "object": row.get(object_col),
                }
                
                if confidence_col and confidence_col in row:
                    try:
                        data["confidence"] = float(row[confidence_col])
                    except (ValueError, TypeError):
                        pass
                
                triple = self.parse_triple(data)
                if triple:
                    mmkg.add(triple)
        
        return mmkg
    
    def load_from_triples(
        self,
        triples: List[Tuple[str, str, str]],
        confidence: Optional[float] = None,
        source: Optional[str] = None,
    ) -> MMKG:
        """
        Load MMKG from a list of (subject, relation, object) tuples.
        
        Args:
            triples: List of (s, r, o) tuples
            confidence: Confidence score for all triples
            source: Source label for all triples
            
        Returns:
            MMKG populated with triples
        """
        mmkg = MMKG(confidence_threshold=self.confidence_threshold)
        
        for s, r, o in triples:
            triple = Triple(
                subject=str(s).strip(),
                relation=str(r).strip(),
                object=str(o).strip(),
                confidence=confidence or self.default_confidence,
                source=source or self.default_source,
            )
            mmkg.add(triple)
        
        return mmkg
    
    def iter_qa_pairs(
        self,
        path: str,
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over QA pairs from a dataset file.
        
        Yields dictionaries with keys: question, image_path, answer, etc.
        
        Args:
            path: Path to dataset file (JSON or JSONL)
            
        Yields:
            QA pair dictionaries
        """
        path = Path(path)
        
        if path.suffix == ".jsonl":
            with open(path, "r") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        else:
            with open(path, "r") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                yield from data
            elif "data" in data:
                yield from data["data"]
            elif "questions" in data:
                yield from data["questions"]


def load_mmkg_from_dataset(
    path: str,
    format: str = "json",
    **kwargs: Any,
) -> MMKG:
    """
    Convenience function to load MMKG from a dataset file.
    
    Args:
        path: Path to dataset file
        format: File format ('json', 'csv')
        **kwargs: Additional arguments for loader
        
    Returns:
        MMKG populated with dataset triples
    """
    loader = DatasetLoader(**kwargs)
    
    if format == "json":
        return loader.load_from_json(path)
    elif format == "csv":
        return loader.load_from_csv(path)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'csv'.")


def create_sample_mmkg() -> MMKG:
    """
    Create a sample MMKG for testing and demonstration.
    
    Returns:
        MMKG with sample triples
    """
    sample_triples = [
        ("Sun", "is_a", "star"),
        ("Sun", "located_in", "Solar System"),
        ("Earth", "orbits", "Sun"),
        ("Earth", "is_a", "planet"),
        ("Moon", "orbits", "Earth"),
        ("Moon", "is_a", "satellite"),
        ("Mars", "orbits", "Sun"),
        ("Mars", "is_a", "planet"),
        ("Jupiter", "orbits", "Sun"),
        ("Jupiter", "is_a", "planet"),
        ("Jupiter", "has", "Great Red Spot"),
        ("Solar System", "contains", "Sun"),
        ("Solar System", "contains", "Earth"),
        ("Solar System", "contains", "Mars"),
        ("Solar System", "located_in", "Milky Way"),
        ("Milky Way", "is_a", "galaxy"),
        ("Water", "essential_for", "life"),
        ("Earth", "has", "Water"),
        ("Mars", "may_have", "Water"),
        ("Photosynthesis", "requires", "sunlight"),
        ("Plants", "perform", "Photosynthesis"),
        ("Oxygen", "produced_by", "Photosynthesis"),
        ("Humans", "breathe", "Oxygen"),
        ("Humans", "live_on", "Earth"),
    ]
    
    loader = DatasetLoader()
    return loader.load_from_triples(sample_triples, confidence=0.9, source="sample")
