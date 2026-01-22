"""Tests for GraphLoom pipeline."""

import pytest
import numpy as np

from graphloom.core.data_structures import Triple, MMKG, EvidenceSubgraph
from graphloom.phase1.phase1_pipeline import Phase1Pipeline, phase_1_unified
from graphloom.phase2.phase2_pipeline import Phase2Pipeline, phase_2_simplified
from graphloom.phase3.phase3_pipeline import Phase3Pipeline, phase_3_controlled
from graphloom.pipeline import GraphLoomPipeline, answer_question
from graphloom.utils.config import GraphLoomConfig
from graphloom.utils.dataset_loader import create_sample_mmkg


class TestPhase1Pipeline:
    """Tests for Phase 1 pipeline."""
    
    def test_phase1_creation(self):
        pipeline = Phase1Pipeline(encoder_type="mock", extractor_type="mock")
        assert pipeline is not None
    
    def test_phase1_generate_description(self):
        pipeline = Phase1Pipeline(encoder_type="mock", extractor_type="mock")
        desc = pipeline.generate_description("image.jpg", "What is this?")
        assert isinstance(desc, str)
        assert len(desc) > 0
    
    def test_phase1_extract_triples(self):
        pipeline = Phase1Pipeline(encoder_type="mock", extractor_type="mock")
        triples = pipeline.extract_triples("The sun is a star.")
        assert isinstance(triples, list)
    
    def test_phase1_process(self):
        pipeline = Phase1Pipeline(encoder_type="mock", extractor_type="mock")
        mmkg = MMKG()
        
        mmkg, triples = pipeline.process("image.jpg", "What is this?", mmkg)
        
        assert isinstance(mmkg, MMKG)
        assert isinstance(triples, list)
    
    def test_phase1_convenience_function(self):
        mmkg = MMKG()
        mmkg, triples = phase_1_unified(
            "image.jpg", "What is this?", mmkg,
            encoder_type="mock", extractor_type="mock"
        )
        
        assert isinstance(mmkg, MMKG)
        assert isinstance(triples, list)


class TestPhase2Pipeline:
    """Tests for Phase 2 pipeline."""
    
    def test_phase2_creation(self):
        pipeline = Phase2Pipeline(embedder_type="mock")
        assert pipeline is not None
    
    def test_phase2_compute_query_embedding(self):
        pipeline = Phase2Pipeline(embedder_type="mock")
        embedding = pipeline.compute_query_embedding(None, "What is this?")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
    
    def test_phase2_process(self):
        pipeline = Phase2Pipeline(embedder_type="mock")
        mmkg = create_sample_mmkg()
        triples = [Triple(subject="Test", relation="is", object="example")]
        
        subgraph = pipeline.process(mmkg, triples, None, "What orbits the Sun?")
        
        assert isinstance(subgraph, EvidenceSubgraph)
        assert len(subgraph.seed_triples) > 0
    
    def test_phase2_convenience_function(self):
        mmkg = create_sample_mmkg()
        triples = []
        
        subgraph = phase_2_simplified(
            mmkg, triples, None, "What is Earth?",
            embedder_type="mock"
        )
        
        assert isinstance(subgraph, EvidenceSubgraph)


class TestPhase3Pipeline:
    """Tests for Phase 3 pipeline."""
    
    def test_phase3_creation(self):
        pipeline = Phase3Pipeline(generator_type="mock")
        assert pipeline is not None
    
    def test_phase3_encode_slots(self):
        pipeline = Phase3Pipeline(generator_type="mock", embedder_type="mock")
        
        mmkg = MMKG()
        triple = Triple(subject="A", relation="r", object="B")
        mmkg.add(triple)
        
        subgraph = EvidenceSubgraph(
            seed_triples=[triple],
            expanded_triples=[],
            mmkg=mmkg,
        )
        
        slot_bank = pipeline.encode_slots(subgraph)
        assert len(slot_bank) > 0
    
    def test_phase3_process(self):
        pipeline = Phase3Pipeline(generator_type="mock", embedder_type="mock")
        
        mmkg = MMKG()
        triple = Triple(subject="Earth", relation="orbits", object="Sun")
        mmkg.add(triple)
        
        subgraph = EvidenceSubgraph(
            seed_triples=[triple],
            expanded_triples=[],
            mmkg=mmkg,
        )
        
        answer = pipeline.process(subgraph, None, "What does Earth orbit?")
        
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    def test_phase3_convenience_function(self):
        mmkg = MMKG()
        triple = Triple(subject="Earth", relation="orbits", object="Sun")
        mmkg.add(triple)
        
        subgraph = EvidenceSubgraph(
            seed_triples=[triple],
            expanded_triples=[],
            mmkg=mmkg,
        )
        
        answer = phase_3_controlled(
            subgraph, None, "What does Earth orbit?",
            generator_type="mock", embedder_type="mock"
        )
        
        assert isinstance(answer, str)


class TestGraphLoomPipeline:
    """Tests for main GraphLoom pipeline."""
    
    def test_pipeline_creation_default(self):
        pipeline = GraphLoomPipeline()
        assert pipeline is not None
    
    def test_pipeline_creation_with_config(self):
        config = GraphLoomConfig.from_preset("mock")
        pipeline = GraphLoomPipeline(config=config)
        assert pipeline is not None
    
    def test_pipeline_answer_question(self):
        pipeline = GraphLoomPipeline()
        mmkg = create_sample_mmkg()
        
        answer = pipeline.answer_question(
            "image.jpg",
            "What does Earth orbit?",
            mmkg,
        )
        
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    def test_pipeline_answer_question_with_evidence(self):
        pipeline = GraphLoomPipeline()
        mmkg = create_sample_mmkg()
        
        answer, evidence = pipeline.answer_question(
            "image.jpg",
            "What does Earth orbit?",
            mmkg,
            return_evidence=True,
        )
        
        assert isinstance(answer, str)
        assert isinstance(evidence, EvidenceSubgraph)
    
    def test_pipeline_batch_answer(self):
        pipeline = GraphLoomPipeline()
        mmkg = create_sample_mmkg()
        
        qa_pairs = [
            ("image1.jpg", "What is the Sun?"),
            ("image2.jpg", "What does Earth orbit?"),
        ]
        
        answers = pipeline.batch_answer(qa_pairs, mmkg)
        
        assert len(answers) == 2
        assert all(isinstance(a, str) for a in answers)
    
    def test_convenience_function(self):
        mmkg = create_sample_mmkg()
        
        answer = answer_question(
            "image.jpg",
            "What is Earth?",
            mmkg,
        )
        
        assert isinstance(answer, str)


class TestConfig:
    """Tests for configuration."""
    
    def test_config_from_preset_mock(self):
        config = GraphLoomConfig.from_preset("mock")
        assert config.phase1.encoder_type == "mock"
        assert config.phase2.embedder_type == "mock"
        assert config.phase3.generator_type == "mock"
    
    def test_config_from_preset_lightweight(self):
        config = GraphLoomConfig.from_preset("lightweight")
        assert config.phase1.extractor_type == "spacy"
        assert config.phase2.embedder_type == "sentence_transformer"
    
    def test_config_from_preset_production(self):
        config = GraphLoomConfig.from_preset("production")
        assert config.phase1.encoder_type == "qwen3vl"
        assert config.phase1.extractor_type == "rebel"
        assert config.phase3.generator_type == "llama"
    
    def test_config_to_dict(self):
        config = GraphLoomConfig.from_preset("mock")
        config_dict = config.to_dict()
        
        assert "phase1" in config_dict
        assert "phase2" in config_dict
        assert "phase3" in config_dict
    
    def test_config_from_dict(self):
        config_dict = {
            "phase1": {"encoder_type": "mock"},
            "phase2": {"embedder_type": "mock"},
            "phase3": {"generator_type": "mock"},
        }
        
        config = GraphLoomConfig.from_dict(config_dict)
        assert config.phase1.encoder_type == "mock"
