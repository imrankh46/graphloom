# GraphLoom

**Controlled Evidence Injection for Interleaved KG-RAG in Multimodal QA**

GraphLoom is a multimodal KG-RAG framework for grounded multimodal question answering. It unifies retrieval-augmented generation (RAG), instance-level multimodal knowledge graphs, and frozen large language models.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Formats](#dataset-formats)
- [Detailed Usage](#detailed-usage)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)

## Architecture Overview

GraphLoom operates in three phases:

### Phase 1: Unified MMKG Build

This phase processes an image and question to extract knowledge triples and build the MMKG.

```
Input: Image I, Question Q, MMKG
    ↓
[Qwen3-VL-Instruct] → Unified factual description
    ↓
[REBEL Extractor] → Structured triples (subject, relation, object)
    ↓
[MMKG.add()] → Insert triples into knowledge graph
    ↓
Output: Updated MMKG, Extracted triples
```

### Phase 2: Evidence Subgraph Retrieval

This phase retrieves relevant evidence from the MMKG for answering the question.

```
Input: MMKG, Extracted triples, Image I, Question Q
    ↓
[Qwen3-VL-Embedding] → Multimodal query embedding
    ↓
[Multimodal Retrieval] → Candidate triples with scores
    ↓
[Top-K Selection] → Seed triples (k=15 default)
    ↓
[Bounded Graph Expansion] → BFS with degree capping (max_hops=2, max_degree=5)
    ↓
Output: Compact evidence subgraph
```

### Phase 3: Controlled Reasoning

This phase generates the final answer using the evidence subgraph.

```
Input: Evidence subgraph, Image I, Question Q
    ↓
[HieraSlot Encoder] → Hierarchical slot memories (entity + relation slots)
    ↓
[Reliability-Calibrated Router] → Dynamic slot selection (top-k=4)
    ↓
[KG-JSA++ Attention] → Joint graph-sequence attention
    ↓
[Llama-3.1-8B-Instruct] → Faithful answer generation
    ↓
Output: Final answer
```

## Installation

### Basic Installation (Mock Components Only)

```bash
git clone https://github.com/imrankh46/graphloom.git
cd graphloom
pip install -e .
```

### With Lightweight Models

```bash
pip install -e ".[lightweight]"
python -m spacy download en_core_web_sm
```

### Full Installation (GPU Required)

```bash
pip install -e ".[full]"
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from graphloom import GraphLoomPipeline, MMKG
from graphloom.utils import create_sample_mmkg

# Create pipeline with mock components (for testing)
pipeline = GraphLoomPipeline()

# Create a sample MMKG for testing
mmkg = create_sample_mmkg()

# Answer a question
image = "path/to/image.jpg"  # or numpy array, or None for text-only
question = "What does Earth orbit?"
answer = pipeline.answer_question(image, question, mmkg)
print(answer)
```

### With Evidence Inspection

```python
# Get answer with evidence for debugging
answer, evidence = pipeline.answer_question(
    image, question, mmkg, 
    return_evidence=True
)

print(f"Answer: {answer}")
print(f"Evidence triples: {len(evidence.all_triples)}")
for triple in evidence.seed_triples:
    print(f"  - {triple.to_text()} (confidence: {triple.confidence:.2f})")
```

## Dataset Formats

GraphLoom supports multiple dataset formats for loading knowledge graphs.

### JSON Format

**Option 1: List of Triples**

```json
{
  "triples": [
    {
      "subject": "Sun",
      "relation": "is_a",
      "object": "star",
      "confidence": 0.95
    },
    {
      "subject": "Earth",
      "relation": "orbits",
      "object": "Sun",
      "confidence": 0.99
    },
    {
      "subject": "Moon",
      "relation": "orbits",
      "object": "Earth",
      "confidence": 0.98
    }
  ]
}
```

**Option 2: Simple Array**

```json
[
  ["Sun", "is_a", "star"],
  ["Earth", "orbits", "Sun"],
  ["Moon", "orbits", "Earth"]
]
```

**Option 3: Objects with Custom Keys**

```json
[
  {"head": "Sun", "rel": "is_a", "tail": "star"},
  {"head": "Earth", "rel": "orbits", "tail": "Sun"}
]
```

### CSV Format

```csv
subject,relation,object,confidence
Sun,is_a,star,0.95
Earth,orbits,Sun,0.99
Moon,orbits,Earth,0.98
Mars,orbits,Sun,0.97
Jupiter,orbits,Sun,0.96
```

### TSV Format

```
subject	relation	object	confidence
Sun	is_a	star	0.95
Earth	orbits	Sun	0.99
```

### Loading Datasets

```python
from graphloom.utils import DatasetLoader, load_mmkg_from_dataset

# Load from JSON file
mmkg = load_mmkg_from_dataset("knowledge_base.json", format="json")

# Load from CSV file
mmkg = load_mmkg_from_dataset("triples.csv", format="csv")

# Load with custom column names
loader = DatasetLoader()
mmkg = loader.load_from_csv(
    "custom_data.csv",
    subject_col="head",
    relation_col="rel", 
    object_col="tail",
    confidence_col="score"
)

# Load from Python list of tuples
triples = [
    ("Sun", "is_a", "star"),
    ("Earth", "orbits", "Sun"),
    ("Moon", "orbits", "Earth"),
]
mmkg = loader.load_from_triples(triples, confidence=0.9, source="manual")
```

### QA Dataset Format

For batch processing of QA pairs:

```json
{
  "qa_pairs": [
    {
      "image": "images/solar_system.jpg",
      "question": "What does Earth orbit?",
      "answer": "Sun"
    },
    {
      "image": "images/moon.jpg", 
      "question": "What orbits Earth?",
      "answer": "Moon"
    }
  ]
}
```

```python
# Iterate over QA pairs
loader = DatasetLoader()
for qa in loader.iter_qa_pairs("qa_dataset.json"):
    image = qa.get("image")
    question = qa["question"]
    ground_truth = qa.get("answer")
    
    predicted = pipeline.answer_question(image, question, mmkg)
    print(f"Q: {question}")
    print(f"Predicted: {predicted}")
    print(f"Ground Truth: {ground_truth}")
```

## Detailed Usage

### Using Individual Phases

```python
from graphloom.core import MMKG, Triple
from graphloom.phase1 import Phase1Pipeline
from graphloom.phase2 import Phase2Pipeline
from graphloom.phase3 import Phase3Pipeline

# Initialize MMKG
mmkg = MMKG()

# Phase 1: Build MMKG from image and question
phase1 = Phase1Pipeline(encoder_type="mock", extractor_type="mock")
mmkg, extracted_triples = phase1.process(image, question, mmkg)

print(f"Extracted {len(extracted_triples)} triples:")
for t in extracted_triples:
    print(f"  {t.to_text()}")

# Phase 2: Retrieve evidence subgraph
phase2 = Phase2Pipeline(embedder_type="mock")
evidence_subgraph = phase2.process(mmkg, extracted_triples, image, question)

print(f"Evidence subgraph: {len(evidence_subgraph.all_triples)} triples")

# Phase 3: Generate answer
phase3 = Phase3Pipeline(generator_type="mock")
answer = phase3.process(evidence_subgraph, image, question)

print(f"Answer: {answer}")
```

### Building MMKG Manually

```python
from graphloom.core import MMKG, Triple

# Create empty MMKG
mmkg = MMKG(confidence_threshold=0.5)

# Add individual triples
triple = Triple(
    subject="Earth",
    relation="orbits",
    object="Sun",
    confidence=0.99,
    source="astronomy_kb"
)
mmkg.add(triple)

# Add batch of triples
triples = [
    Triple(subject="Moon", relation="orbits", object="Earth", confidence=0.98),
    Triple(subject="Mars", relation="orbits", object="Sun", confidence=0.97),
    Triple(subject="Sun", relation="is_a", object="star", confidence=0.95),
]
added_count = mmkg.add_batch(triples)
print(f"Added {added_count} triples")

# Query the MMKG
neighbors = mmkg.get_neighbors("Sun")
print(f"Neighbors of Sun: {[t.to_text() for t in neighbors]}")

# Get all entities
entities = mmkg.get_entities()
print(f"Entities: {entities}")

# Get subgraph for specific entities
subgraph_triples = mmkg.subgraph_from_entities({"Earth", "Sun", "Moon"})
```

### Batch Processing

```python
# Process multiple QA pairs
qa_pairs = [
    ("image1.jpg", "What does Earth orbit?"),
    ("image2.jpg", "What is the Sun?"),
    ("image3.jpg", "What orbits Earth?"),
]

answers = pipeline.batch_answer(qa_pairs, mmkg)
for (img, q), a in zip(qa_pairs, answers):
    print(f"Q: {q} -> A: {a}")
```

## Configuration

### Configuration Presets

| Preset | Phase 1 | Phase 2 | Phase 3 | Use Case |
|--------|---------|---------|---------|----------|
| `mock` | Mock encoder/extractor | Mock embedder | Mock generator | Testing without models |
| `lightweight` | Mock + spaCy | sentence-transformers | Mock | CPU-only, lightweight |
| `production` | Qwen3-VL + REBEL | Qwen3-VL-Embedding | Llama-3.1 | Full GPU pipeline |

### Using Presets

```python
from graphloom.utils import GraphLoomConfig
from graphloom import GraphLoomPipeline

# Use mock preset for testing
config = GraphLoomConfig.from_preset("mock")
pipeline = GraphLoomPipeline(config=config)

# Use lightweight preset for CPU
config = GraphLoomConfig.from_preset("lightweight")
pipeline = GraphLoomPipeline(config=config)

# Use production preset for full pipeline
config = GraphLoomConfig.from_preset("production")
pipeline = GraphLoomPipeline(config=config)
```

### Custom Configuration

```python
from graphloom.utils import GraphLoomConfig
from graphloom.utils.config import Phase1Config, Phase2Config, Phase3Config

config = GraphLoomConfig(
    phase1=Phase1Config(
        encoder_type="qwen3vl",           # or "mock"
        extractor_type="rebel",            # or "spacy", "mock"
        confidence_threshold=0.5,
        qwen_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        rebel_model_name="Babelscape/rebel-large",
    ),
    phase2=Phase2Config(
        embedder_type="sentence_transformer",  # or "qwen3vl", "mock"
        top_k=15,                              # number of seed triples
        max_hops=2,                            # BFS expansion hops
        max_degree=5,                          # max neighbors per node
        max_edges=50,                          # max total edges in subgraph
        model_name="all-MiniLM-L6-v2",
    ),
    phase3=Phase3Config(
        generator_type="llama",           # or "mock"
        top_k_slots=4,                    # slots for attention
        gate_threshold=0.3,               # routing threshold
        reliability_weight=2.0,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        hidden_dim=4096,
    ),
)

pipeline = GraphLoomPipeline(config=config)
```

### Save and Load Configuration

```python
# Save configuration to JSON
config.save_json("my_config.json")

# Load configuration from JSON
config = GraphLoomConfig.from_json("my_config.json")

# Convert to dictionary
config_dict = config.to_dict()
```

### Example Configuration File

```json
{
  "phase1": {
    "encoder_type": "mock",
    "extractor_type": "spacy",
    "confidence_threshold": 0.5
  },
  "phase2": {
    "embedder_type": "sentence_transformer",
    "top_k": 15,
    "max_hops": 2,
    "max_degree": 5,
    "max_edges": 50
  },
  "phase3": {
    "generator_type": "mock",
    "top_k_slots": 4,
    "gate_threshold": 0.3
  }
}
```

## API Reference

### Core Data Structures

#### Triple

```python
from graphloom.core import Triple

triple = Triple(
    subject="Earth",           # Subject entity
    relation="orbits",         # Relation type
    object="Sun",              # Object entity
    confidence=0.99,           # Confidence score [0, 1]
    source="astronomy_kb",     # Optional source identifier
)

# Methods
triple.to_text()      # Returns "Earth orbits Sun"
triple.to_tuple()     # Returns ("Earth", "orbits", "Sun")
```

#### MMKG

```python
from graphloom.core import MMKG

mmkg = MMKG(confidence_threshold=0.0)

# Methods
mmkg.add(triple)                          # Add single triple
mmkg.add_batch(triples)                   # Add multiple triples
mmkg.get_neighbors(entity, max_degree=5)  # Get neighboring triples
mmkg.get_entities()                       # Get all entities
mmkg.subgraph_from_entities(entities)     # Extract subgraph
mmkg.get_all_triples()                    # Get all triples
len(mmkg)                                 # Number of triples
triple in mmkg                            # Check membership
```

#### EvidenceSubgraph

```python
from graphloom.core import EvidenceSubgraph

# Properties
subgraph.seed_triples      # Initial retrieved triples
subgraph.expanded_triples  # Triples from graph expansion
subgraph.all_triples       # All triples combined
subgraph.mmkg              # Reference to source MMKG

# Methods
subgraph.to_text()         # Text representation of evidence
```

### Main Pipeline

```python
from graphloom import GraphLoomPipeline

pipeline = GraphLoomPipeline(config=None)

# Answer a single question
answer = pipeline.answer_question(
    image,                    # str path, numpy array, or None
    question,                 # str
    mmkg,                     # MMKG instance
    return_evidence=False,    # If True, returns (answer, evidence)
)

# Batch processing
answers = pipeline.batch_answer(
    qa_pairs,                 # List of (image, question) tuples
    mmkg,                     # MMKG instance
)
```

### Convenience Functions

```python
# Phase 1
from graphloom.phase1 import phase_1_unified
mmkg, triples = phase_1_unified(image, question, mmkg)

# Phase 2
from graphloom.phase2 import phase_2_simplified
subgraph = phase_2_simplified(mmkg, triples, image, question)

# Phase 3
from graphloom.phase3 import phase_3_controlled
answer = phase_3_controlled(subgraph, image, question)

# Full pipeline
from graphloom import answer_question
answer = answer_question(image, question, mmkg)
```

## Project Structure

```
graphloom/
├── graphloom/
│   ├── __init__.py              # Package exports
│   ├── pipeline.py              # Main GraphLoomPipeline
│   ├── core/
│   │   ├── __init__.py
│   │   └── data_structures.py   # Triple, MMKG, HieraSlot, etc.
│   ├── phase1/
│   │   ├── __init__.py
│   │   ├── multimodal_encoder.py    # Qwen3-VL encoder
│   │   ├── triple_extractor.py      # REBEL, spaCy extractors
│   │   └── phase1_pipeline.py       # Phase 1 orchestration
│   ├── phase2/
│   │   ├── __init__.py
│   │   ├── multimodal_embedder.py   # Embedding models
│   │   ├── retrieval.py             # Evidence retrieval
│   │   └── phase2_pipeline.py       # Phase 2 orchestration
│   ├── phase3/
│   │   ├── __init__.py
│   │   ├── hieraslot.py             # HieraSlot encoder
│   │   ├── router.py                # Reliability-calibrated router
│   │   ├── kg_jsa.py                # KG-JSA++ attention
│   │   ├── answer_generator.py      # Llama answer generation
│   │   └── phase3_pipeline.py       # Phase 3 orchestration
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # Configuration classes
│       └── dataset_loader.py        # Dataset loading utilities
├── tests/
│   ├── test_core.py                 # Core data structure tests
│   └── test_pipeline.py             # Pipeline tests
├── pyproject.toml                   # Project configuration
└── README.md                        # This file
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=graphloom --cov-report=html
```

## Design Decisions

This implementation follows a **simplified architecture** that:

1. **Skips enrichment**: No external KG lookups (VG150, ConceptNet) - can be added later
2. **No canonicalization**: Triples inserted directly without entity linking
3. **Modular design**: Each phase is independent and can be customized
4. **Mock components**: Test the full pipeline without GPU/model dependencies
5. **Extensible**: Abstract base classes allow easy addition of new encoders, extractors, etc.

## License

MIT License
