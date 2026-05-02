# Ablation Framework How-To Guide

This guide explains how to use the ablation framework to run systematic experiments comparing different pipeline configurations for the Omnilex Legal Retrieval competition.

## Overview

The ablation framework allows you to:
- Run experiments with different component combinations
- Measure each component's contribution to Macro F1 score
- Compare results across different configurations
- Export results for analysis

## Quick Start

### Running a Single Experiment

```python
from omnilex.retrieval.ablation.config import ExperimentConfig
from omnilex.retrieval.ablation.runner import ExperimentRunner

# Load a preset configuration
config = ExperimentConfig.from_preset('exp_full_pipeline')

# Create runner
runner = ExperimentRunner(config, output_dir='experiments')

# Run experiment
queries = [
    {'id': 'q1', 'query': 'What are the requirements for a valid contract?'},
    # ... more queries
]
results = runner.run(queries)
print(f"Macro F1: {results['metrics']['macro_f1']}")
```

### Running All Presets

```python
from omnilex.retrieval.ablation.config import EXPERIMENT_PRESETS
from omnilex.retrieval.ablation.runner import ExperimentRunner
from omnilex.retrieval.ablation.reporter import ResultsReporter

reporter = ResultsReporter('experiments')

for preset_name in EXPERIMENT_PRESETS.keys():
    print(f"Running {preset_name}...")
    config = ExperimentConfig.from_preset(preset_name)
    runner = ExperimentRunner(config, output_dir='experiments')
    results = runner.run(queries, ground_truth)
    
    # Add to comparison
    reporter.add_result(
        preset_name,
        config.to_dict(),
        results['metrics'],
        per_query_f1=results.get('per_query_f1', [])
    )

# Print comparison table
reporter.print_summary()
reporter.export_csv('ablation_results.csv')
```

## Experiment Presets

The framework includes 7 predefined presets:

| Preset Name | BM25 | Dense | Graph | RRF | Reranker | Verifier | Expected F1 |
|------------|------|-------|-------|-----|----------|----------|------------|
| `exp_baseline` | ✓ | | | | | | ~0.327 |
| `exp_dense_only` | | ✓ | | | | | ~0.489 |
| `exp_bm25_dense` | ✓ | ✓ | | ✓ | | | ~0.500 |
| `exp_full_retrieval` | ✓ | ✓ | ✓ | ✓ | | | ~0.543 |
| `exp_full_rrf` | ✓ | ✓ | ✓ | ✓ | | | ~0.600 |
| `exp_full_reranker` | ✓ | ✓ | ✓ | ✓ | ✓ | | ~0.650 |
| `exp_full_pipeline` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ~0.691 |

## Custom Configuration

You can create custom configurations using YAML or Python:

### YAML Configuration

Create a file `my_experiment.yaml`:

```yaml
name: my_experiment
description: "BM25 + Dense with custom weights"
components:
  bm25: true
  dense: true
  graph: false
  rrf_fusion: true
  reranker: false
  verifier: false
signal_weights:
  bm25: 0.3
  dense: 0.7
dense_index_preset: balanced
top_k: 50
fusion_top_k: 20
reranker_top_k: 10
verifier_threshold: 0.5
```

Load and run:

```python
config = ExperimentConfig.from_yaml('my_experiment.yaml')
runner = ExperimentRunner(config)
results = runner.run(queries)
```

### Python Configuration

```python
config = ExperimentConfig(
    name='custom_experiment',
    description='Custom configuration',
    components={
        'bm25': True,
        'dense': True,
        'graph': False,
        'rrf_fusion': True,
        'reranker': False,
        'verifier': False,
    },
    signal_weights={'bm25': 0.4, 'dense': 0.6},
    dense_index_preset='quality',
    top_k=100,
)
```

## Component Details

### BM25 Retrieval
- Uses `rank-bm25` for keyword search
- Fast, no GPU required
- Good for exact keyword matching

### Dense Retrieval (BGE-M3 + FAISS)
- Uses BGE-M3 multilingual embeddings (1024 dimensions)
- FAISS for fast similarity search
- Three index presets:
  - `quality`: IndexFlatIP (exact, ~10GB RAM)
  - `balanced`: IndexHNSWFlat (M=16, ~4-6GB RAM)
  - `minimal`: IndexIVFPQ (quantized, ~1.2GB RAM)

### Citation Graph
- Builds knowledge graph from citation patterns
- Uses Personalized PageRank (PPR) for ranking
- Leiden community detection for clustering
- Co-citation analysis and bibliographic coupling

### Signal Fusion (Weighted RRF)
- Combines multiple retrieval signals
- Reciprocal Rank Fusion with configurable weights
- Cross-signal boosting for documents in multiple signals

### Reranker (BGE-reranker-v2-m3)
- Cross-encoder reranking
- Refines top candidates after fusion
- Score normalization to [0, 1]

### LLM Verifier (Qwen2.5-7B)
- Verification-only (never generates citations)
- Scores candidate citations 0-1
- Threshold filtering (default 0.5)
- Fallback to heuristic if model unavailable

## Evaluating Results

### Validation

After each experiment, validate your submission:

```bash
python scripts/validate_submission.py submission.csv
```

### Metrics Tracked

The framework tracks component-level metrics:

- **Retrieval**: Recall@5, Recall@10, Recall@20, Recall@50
- **Reranker**: Accuracy@K, NDCG
- **Verifier**: Precision, Recall, F1

### Exporting Results

```python
# Export to CSV
reporter.export_csv('results.csv')

# Export to JSON
reporter.export_json('results.json')

# Print comparison table
print(reporter.generate_comparison_table())
```

## Notebook Examples

See the notebooks directory for examples:

1. **02_agentic_retrieval_baseline.ipynb** - Baseline with ablation section
2. **03_hybrid_graphrag_pipeline.ipynb** - Full SOTA pipeline

## Tips for Kaggle

1. **Memory constraints**: Use `minimal` preset for FAISS if <4GB VRAM
2. **Model files**: Upload GGUF models as Kaggle dataset
3. **Pre-built indices**: Cache FAISS indices as Kaggle dataset
4. **Submission**: Run `python scripts/validate_submission.py` before submitting

## Troubleshooting

### ImportError: No module named 'faiss'
```bash
pip install faiss-cpu  # or faiss-gpu
```

### ImportError: No module named 'llama_cpp'
```bash
pip install llama-cpp-python
```

### Out of Memory
- Use `dense_index_preset: minimal`
- Reduce `top_k` parameter
- Use quantized GGUF models

### Low F1 Score
- Check citation normalization is working
- Verify graph construction is extracting citations
- Increase `top_k` for retrieval
- Lower `verifier_threshold` to keep more candidates
