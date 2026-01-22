"""
Configuration management for GraphLoom.

This module provides centralized configuration for all GraphLoom components.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class Phase1Config:
    """Configuration for Phase 1: MMKG Build."""
    encoder_type: str = "mock"
    extractor_type: str = "mock"
    confidence_threshold: float = 0.5
    encoder_kwargs: Dict[str, Any] = field(default_factory=dict)
    extractor_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Phase2Config:
    """Configuration for Phase 2: Evidence Retrieval."""
    embedder_type: str = "mock"
    top_k: int = 15
    max_hops: int = 2
    max_degree: int = 5
    max_edges: int = 50
    similarity_threshold: float = 0.3
    embedder_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Phase3Config:
    """Configuration for Phase 3: Controlled Reasoning."""
    generator_type: str = "mock"
    hidden_dim: int = 4096
    top_k_slots: int = 4
    gate_threshold: float = 0.3
    reliability_weight: float = 2.0
    prefix_length: int = 32
    recent_tokens: int = 64
    generator_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphLoomConfig:
    """
    Main configuration for GraphLoom pipeline.
    
    Example:
        >>> config = GraphLoomConfig.from_preset("production")
        >>> pipeline = GraphLoomPipeline(config=config)
    """
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    
    mmkg_confidence_threshold: float = 0.0
    device: str = "auto"
    verbose: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GraphLoomConfig":
        """Create config from dictionary."""
        phase1_dict = config_dict.get("phase1", {})
        phase2_dict = config_dict.get("phase2", {})
        phase3_dict = config_dict.get("phase3", {})
        
        return cls(
            phase1=Phase1Config(**phase1_dict),
            phase2=Phase2Config(**phase2_dict),
            phase3=Phase3Config(**phase3_dict),
            mmkg_confidence_threshold=config_dict.get("mmkg_confidence_threshold", 0.0),
            device=config_dict.get("device", "auto"),
            verbose=config_dict.get("verbose", False),
        )
    
    @classmethod
    def from_json(cls, path: str) -> "GraphLoomConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_preset(cls, preset: str) -> "GraphLoomConfig":
        """
        Create config from a preset.
        
        Available presets:
        - "mock": For testing without GPU/models
        - "lightweight": Uses smaller models (sentence-transformers, spacy)
        - "production": Full models (Qwen3-VL, REBEL, Llama)
        """
        presets = {
            "mock": cls._mock_preset(),
            "lightweight": cls._lightweight_preset(),
            "production": cls._production_preset(),
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. "
                           f"Available: {list(presets.keys())}")
        
        return presets[preset]
    
    @classmethod
    def _mock_preset(cls) -> "GraphLoomConfig":
        """Mock preset for testing."""
        return cls(
            phase1=Phase1Config(
                encoder_type="mock",
                extractor_type="mock",
            ),
            phase2=Phase2Config(
                embedder_type="mock",
            ),
            phase3=Phase3Config(
                generator_type="mock",
            ),
        )
    
    @classmethod
    def _lightweight_preset(cls) -> "GraphLoomConfig":
        """Lightweight preset using smaller models."""
        return cls(
            phase1=Phase1Config(
                encoder_type="mock",
                extractor_type="spacy",
                extractor_kwargs={"model_name": "en_core_web_sm"},
            ),
            phase2=Phase2Config(
                embedder_type="sentence_transformer",
                embedder_kwargs={"model_name": "all-MiniLM-L6-v2"},
            ),
            phase3=Phase3Config(
                generator_type="mock",
            ),
        )
    
    @classmethod
    def _production_preset(cls) -> "GraphLoomConfig":
        """Production preset using full models."""
        return cls(
            phase1=Phase1Config(
                encoder_type="qwen3vl",
                extractor_type="rebel",
                encoder_kwargs={"model_name": "Qwen/Qwen2.5-VL-7B-Instruct"},
                extractor_kwargs={"model_name": "Babelscape/rebel-large"},
            ),
            phase2=Phase2Config(
                embedder_type="qwen3vl",
                embedder_kwargs={"model_name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct"},
            ),
            phase3=Phase3Config(
                generator_type="llama",
                generator_kwargs={"model_name": "meta-llama/Llama-3.1-8B-Instruct"},
            ),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "phase1": {
                "encoder_type": self.phase1.encoder_type,
                "extractor_type": self.phase1.extractor_type,
                "confidence_threshold": self.phase1.confidence_threshold,
                "encoder_kwargs": self.phase1.encoder_kwargs,
                "extractor_kwargs": self.phase1.extractor_kwargs,
            },
            "phase2": {
                "embedder_type": self.phase2.embedder_type,
                "top_k": self.phase2.top_k,
                "max_hops": self.phase2.max_hops,
                "max_degree": self.phase2.max_degree,
                "max_edges": self.phase2.max_edges,
                "similarity_threshold": self.phase2.similarity_threshold,
                "embedder_kwargs": self.phase2.embedder_kwargs,
            },
            "phase3": {
                "generator_type": self.phase3.generator_type,
                "hidden_dim": self.phase3.hidden_dim,
                "top_k_slots": self.phase3.top_k_slots,
                "gate_threshold": self.phase3.gate_threshold,
                "reliability_weight": self.phase3.reliability_weight,
                "prefix_length": self.phase3.prefix_length,
                "recent_tokens": self.phase3.recent_tokens,
                "generator_kwargs": self.phase3.generator_kwargs,
            },
            "mmkg_confidence_threshold": self.mmkg_confidence_threshold,
            "device": self.device,
            "verbose": self.verbose,
        }
    
    def save_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
