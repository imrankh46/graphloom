"""
GraphLoom: Controlled Evidence Injection for Interleaved KG-RAG in Multimodal QA

A multimodal KG-RAG framework that unifies retrieval-augmented generation,
instance-level multimodal knowledge graphs, and frozen large language models.
"""

__version__ = "0.1.0"
__author__ = "GraphLoom Team"

from graphloom.core.data_structures import Triple, MMKG, HieraSlot, SlotBank
from graphloom.pipeline import GraphLoomPipeline

__all__ = [
    "Triple",
    "MMKG", 
    "HieraSlot",
    "SlotBank",
    "GraphLoomPipeline",
]
