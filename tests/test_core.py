"""Tests for core data structures."""

import pytest
import numpy as np

from graphloom.core.data_structures import (
    Triple,
    MMKG,
    HieraSlot,
    SlotBank,
    SlotType,
    EvidenceSubgraph,
    RetrievalConfig,
    RouterConfig,
)


class TestTriple:
    """Tests for Triple data structure."""
    
    def test_triple_creation(self):
        triple = Triple(
            subject="Earth",
            relation="orbits",
            object="Sun",
            confidence=0.9,
            source="test",
        )
        assert triple.subject == "Earth"
        assert triple.relation == "orbits"
        assert triple.object == "Sun"
        assert triple.confidence == 0.9
        assert triple.source == "test"
    
    def test_triple_to_text(self):
        triple = Triple(subject="Earth", relation="orbits", object="Sun")
        assert triple.to_text() == "Earth orbits Sun"
    
    def test_triple_to_tuple(self):
        triple = Triple(subject="Earth", relation="orbits", object="Sun")
        assert triple.to_tuple() == ("Earth", "orbits", "Sun")
    
    def test_triple_equality(self):
        t1 = Triple(subject="A", relation="r", object="B")
        t2 = Triple(subject="A", relation="r", object="B")
        t3 = Triple(subject="A", relation="r", object="C")
        
        assert t1 == t2
        assert t1 != t3
    
    def test_triple_hash(self):
        t1 = Triple(subject="A", relation="r", object="B")
        t2 = Triple(subject="A", relation="r", object="B")
        
        assert hash(t1) == hash(t2)
        assert len({t1, t2}) == 1


class TestMMKG:
    """Tests for MMKG data structure."""
    
    def test_mmkg_creation(self):
        mmkg = MMKG()
        assert len(mmkg) == 0
    
    def test_mmkg_add_triple(self):
        mmkg = MMKG()
        triple = Triple(subject="A", relation="r", object="B")
        
        result = mmkg.add(triple)
        assert result is True
        assert len(mmkg) == 1
        assert triple in mmkg
    
    def test_mmkg_add_duplicate(self):
        mmkg = MMKG()
        triple = Triple(subject="A", relation="r", object="B")
        
        mmkg.add(triple)
        result = mmkg.add(triple)
        
        assert result is False
        assert len(mmkg) == 1
    
    def test_mmkg_confidence_threshold(self):
        mmkg = MMKG(confidence_threshold=0.5)
        
        high_conf = Triple(subject="A", relation="r", object="B", confidence=0.8)
        low_conf = Triple(subject="C", relation="r", object="D", confidence=0.3)
        
        assert mmkg.add(high_conf) is True
        assert mmkg.add(low_conf) is False
        assert len(mmkg) == 1
    
    def test_mmkg_add_batch(self):
        mmkg = MMKG()
        triples = [
            Triple(subject="A", relation="r1", object="B"),
            Triple(subject="B", relation="r2", object="C"),
            Triple(subject="A", relation="r1", object="B"),  # duplicate
        ]
        
        added = mmkg.add_batch(triples)
        assert added == 2
        assert len(mmkg) == 2
    
    def test_mmkg_get_neighbors(self):
        mmkg = MMKG()
        t1 = Triple(subject="A", relation="r1", object="B")
        t2 = Triple(subject="A", relation="r2", object="C")
        t3 = Triple(subject="B", relation="r3", object="D")
        
        mmkg.add_batch([t1, t2, t3])
        
        neighbors_a = mmkg.get_neighbors("A")
        assert len(neighbors_a) == 2
        assert t1 in neighbors_a
        assert t2 in neighbors_a
        
        neighbors_b = mmkg.get_neighbors("B")
        assert len(neighbors_b) == 2
        assert t1 in neighbors_b
        assert t3 in neighbors_b
    
    def test_mmkg_get_neighbors_max_degree(self):
        mmkg = MMKG()
        for i in range(10):
            mmkg.add(Triple(subject="A", relation=f"r{i}", object=f"B{i}", confidence=i/10))
        
        neighbors = mmkg.get_neighbors("A", max_degree=3)
        assert len(neighbors) == 3
    
    def test_mmkg_get_entities(self):
        mmkg = MMKG()
        mmkg.add(Triple(subject="A", relation="r", object="B"))
        mmkg.add(Triple(subject="B", relation="r", object="C"))
        
        entities = mmkg.get_entities()
        assert entities == {"A", "B", "C"}
    
    def test_mmkg_subgraph_from_entities(self):
        mmkg = MMKG()
        t1 = Triple(subject="A", relation="r1", object="B")
        t2 = Triple(subject="B", relation="r2", object="C")
        t3 = Triple(subject="C", relation="r3", object="D")
        
        mmkg.add_batch([t1, t2, t3])
        
        subgraph = mmkg.subgraph_from_entities({"A", "B", "C"})
        assert len(subgraph) == 2
        assert t1 in subgraph
        assert t2 in subgraph
        assert t3 not in subgraph


class TestHieraSlot:
    """Tests for HieraSlot data structure."""
    
    def test_hieraslot_creation(self):
        triple = Triple(subject="A", relation="r", object="B")
        slot = HieraSlot(
            slot_type=SlotType.ENTITY,
            key=np.zeros(10),
            value=np.ones(10),
            source_triple=triple,
            reliability_score=0.8,
            retrieval_score=0.7,
        )
        
        assert slot.slot_type == SlotType.ENTITY
        assert slot.reliability_score == 0.8
        assert slot.retrieval_score == 0.7
        assert slot.triple_text == "A r B"


class TestSlotBank:
    """Tests for SlotBank data structure."""
    
    def test_slotbank_creation(self):
        bank = SlotBank()
        assert len(bank) == 0
        assert len(bank.entity_slots) == 0
        assert len(bank.relation_slots) == 0
    
    def test_slotbank_add_slots(self):
        bank = SlotBank()
        triple = Triple(subject="A", relation="r", object="B")
        
        bank.add_slots_from_triple(
            triple=triple,
            key_entity=np.zeros(10),
            value_entity=np.ones(10),
            key_relation=np.zeros(10),
            value_relation=np.ones(10),
        )
        
        assert len(bank) == 2
        assert len(bank.entity_slots) == 1
        assert len(bank.relation_slots) == 1
    
    def test_slotbank_all_slots(self):
        bank = SlotBank()
        triple = Triple(subject="A", relation="r", object="B")
        
        bank.add_slots_from_triple(
            triple=triple,
            key_entity=np.zeros(10),
            value_entity=np.ones(10),
            key_relation=np.zeros(10),
            value_relation=np.ones(10),
        )
        
        all_slots = bank.all_slots
        assert len(all_slots) == 2


class TestEvidenceSubgraph:
    """Tests for EvidenceSubgraph data structure."""
    
    def test_evidence_subgraph_creation(self):
        seed = [Triple(subject="A", relation="r", object="B")]
        expanded = [Triple(subject="B", relation="r", object="C")]
        mmkg = MMKG()
        mmkg.add_batch(seed + expanded)
        
        subgraph = EvidenceSubgraph(
            seed_triples=seed,
            expanded_triples=expanded,
            mmkg=mmkg,
        )
        
        assert len(subgraph.all_triples) == 2
    
    def test_evidence_subgraph_to_text(self):
        seed = [Triple(subject="A", relation="r", object="B")]
        mmkg = MMKG()
        mmkg.add_batch(seed)
        
        subgraph = EvidenceSubgraph(
            seed_triples=seed,
            expanded_triples=[],
            mmkg=mmkg,
        )
        
        text = subgraph.to_text()
        assert "A r B" in text


class TestConfigs:
    """Tests for configuration dataclasses."""
    
    def test_retrieval_config_defaults(self):
        config = RetrievalConfig()
        assert config.top_k == 15
        assert config.max_hops == 2
        assert config.max_degree == 5
        assert config.max_edges == 50
    
    def test_router_config_defaults(self):
        config = RouterConfig()
        assert config.top_k_slots == 4
        assert config.gate_threshold == 0.3
        assert config.reliability_weight == 2.0
