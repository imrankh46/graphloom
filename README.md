# GraphLoom

**Controlled Evidence Injection for Interleaved KG-RAG in Multimodal QA**

GraphLoom is a multimodal KG-RAG framework for grounded multimodal question answering. It unifies retrieval-augmented generation (RAG), instance-level multimodal knowledge graphs, and frozen large language models.

## Architecture Overview

GraphLoom operates in three phases:

### Phase 1: Unified MMKG Build
- Generate unified factual description using Qwen3-VL-Instruct
- Extract structured triples using REBEL
- Insert triples directly into MMKG (simplified, no enrichment)

### Phase 2: Evidence Subgraph Retrieval
- Compute multimodal query embedding using Qwen3-VL-Embedding
- Score triples with unified similarity
- Select top-k seed triples
- Perform bounded graph expansion (BFS with degree capping)

### Phase 3: Controlled Reasoning
- Encode evidence into HieraSlot memories (entity + relation slots)
- Route slots using reliability-calibrated router
- Apply KG-JSA++ joint graph-sequence attention
- Generate faithful answer with frozen Llama decoder

## Installation

```bash
# Basic installation (mock components only)
pip install -e .

# With lightweight models (sentence-transformers, spacy)
pip install -e ".[lightweight]"

# Full installation with all models
pip install -e ".[full]"

# Development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from graphloom import GraphLoomPipeline, MMKG
from graphloom.utils import GraphLoomConfig, create_sample_mmkg

# Create pipeline with mock components (for testing)
pipeline = GraphLoomPipeline()

# Or use a preset configuration
config = GraphLoomConfig.from_preset("mock")  # or "lightweight" or "production"
pipeline = GraphLoomPipeline(config=config)

# Create or load an MMKG
mmkg = create_sample_mmkg()  # Sample MMKG for testing
# Or load from file:
# from graphloom.utils import load_mmkg_from_dataset
# mmkg = load_mmkg_from_dataset("triples.json")

# Answer a question
image = "path/to/image.jpg"  # or numpy array
question = "What planet does Earth orbit?"
answer = pipeline.answer_question(image, question, mmkg)
print(answer)
```

## Detailed Usage

### Using Individual Phases

```python
from graphloom.core import MMKG, Triple
from graphloom.phase1 import Phase1Pipeline
from graphloom.phase2 import Phase2Pipeline
from graphloom.phase3 import Phase3Pipeline

# Phase 1: Build MMKG
phase1 = Phase1Pipeline(encoder_type="mock", extractor_type="mock")
mmkg = MMKG()
mmkg, extracted_triples = phase1.process(image, question, mmkg)

# Phase 2: Retrieve evidence
phase2 = Phase2Pipeline(embedder_type="mock")
evidence_subgraph = phase2.process(mmkg, extracted_triples, image, question)

# Phase 3: Generate answer
phase3 = Phase3Pipeline(generator_type="mock")
answer = phase3.process(evidence_subgraph, image, question)
```

### Loading Data

```python
from graphloom.utils import DatasetLoader, load_mmkg_from_dataset

# From JSON
mmkg = load_mmkg_from_dataset("triples.json", format="json")

# From CSV
mmkg = load_mmkg_from_dataset("triples.csv", format="csv")

# From list of tuples
loader = DatasetLoader()
triples = [
    ("Sun", "is_a", "star"),
    ("Earth", "orbits", "Sun"),
]
mmkg = loader.load_from_triples(triples)
```

### Configuration

```python
from graphloom.utils import GraphLoomConfig

# From preset
config = GraphLoomConfig.from_preset("production")

# From JSON file
config = GraphLoomConfig.from_json("config.json")

# Custom configuration
from graphloom.utils.config import Phase1Config, Phase2Config, Phase3Config

config = GraphLoomConfig(
    phase1=Phase1Config(
        encoder_type="qwen3vl",
        extractor_type="rebel",
        confidence_threshold=0.5,
    ),
    phase2=Phase2Config(
        embedder_type="sentence_transformer",
        top_k=15,
        max_hops=2,
    ),
    phase3=Phase3Config(
        generator_type="llama",
        top_k_slots=4,
    ),
)

# Save configuration
config.save_json("my_config.json")
```

## Core Components

### Data Structures

- **Triple**: Knowledge graph triple (subject, relation, object) with confidence
- **MMKG**: Multimodal Knowledge Graph with efficient entity lookup
- **HieraSlot**: Hierarchical slot memory (entity and relation slots)
- **SlotBank**: Collection of HieraSlot memories
- **EvidenceSubgraph**: Retrieved compact evidence subgraph

### Phase 1 Components

- **MultimodalEncoder**: Qwen3-VL-Instruct for scene description
- **TripleExtractor**: REBEL for structured triple extraction

### Phase 2 Components

- **MultimodalEmbedder**: Qwen3-VL-Embedding for unified embeddings
- **EvidenceRetriever**: Dense retrieval with bounded expansion

### Phase 3 Components

- **HieraSlotEncoder**: Convert triples to slot memories
- **ReliabilityCalibratedRouter**: Dynamic slot routing
- **KGJSAPlusPlus**: Joint graph-sequence attention
- **AnswerGenerator**: Llama-based faithful answer generation

## Configuration Presets

| Preset | Phase 1 | Phase 2 | Phase 3 | Use Case |
|--------|---------|---------|---------|----------|
| `mock` | Mock encoder/extractor | Mock embedder | Mock generator | Testing |
| `lightweight` | Mock + spaCy | sentence-transformers | Mock | CPU-only |
| `production` | Qwen3-VL + REBEL | Qwen3-VL-Embedding | Llama-3.1 | Full pipeline |

## API Reference

### Main Pipeline

```python
class GraphLoomPipeline:
    def answer_question(
        self,
        image: Union[str, Path, np.ndarray],
        question: str,
        mmkg: MMKG,
        return_evidence: bool = False,
    ) -> Union[str, Tuple[str, EvidenceSubgraph]]:
        """Answer a question using the full GraphLoom pipeline."""
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

## Design Decisions

This implementation follows a **simplified architecture** that:

1. **Skips enrichment**: No external KG lookups (VG150, ConceptNet)
2. **No canonicalization**: Triples inserted directly without entity linking
3. **Modular design**: Easy to add enrichment later
4. **Mock components**: Test without GPU/model dependencies

## License

MIT License

# graphloom