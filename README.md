# Omnilex Agentic Retrieval Competition Starter Repo

Official starter repo for Kaggle competiton https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/host/launch-checklist

## Quick Start

### Installation

(Tested with Ubuntu-24.04 in WSL)

```bash
# Clone the repository
git clone https://github.com/Omnilex-AI/Omnilex-Agentic-Retrieval-Competition.git
cd Omnilex-Agentic-Retrieval-Competition

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for testing/linting

# Install package in development mode
pip install -e .
```

### Download Data

Get it from Kaggle into `data` directory

### Run Baselines

Two baseline notebooks are provided:

1. **Direct Generation** (`notebooks/01_direct_generation_baseline.ipynb`)
   - Prompts LLM to directly generate citations
   - Simple but prone to hallucination

2. **Agentic Retrieval** (`notebooks/02_agentic_retrieval_baseline.ipynb`)
   - Uses ReAct-style agent with search tools
   - Grounded in actual legal documents

3. **Hybrid GraphRAG Pipeline** (`notebooks/03_hybrid_graphrag_pipeline.ipynb`)
   - Full SOTA pipeline combining BM25 + Dense + Graph + Reranking + Verification
   - Achieves ~0.691 Macro F1 on validation set
   - See [Ablation How-To](ablation_howto.md) for running experiments

### Ablation Framework

The ablation framework (`src/omnilex/retrieval/ablation/`) allows systematic comparison of pipeline components:

```bash
# Run a specific experiment
python -c "from omnilex.retrieval.ablation.config import ExperimentConfig; \
    from omnilex.retrieval.ablation.runner import ExperimentRunner; \
    config = ExperimentConfig.from_preset('exp_full_pipeline'); \
    runner = ExperimentRunner(config); \
    print('Running experiment...')"

# Or use the ablation framework programmatically
```

See [ablation_howto.md](ablation_howto.md) for detailed examples.

Both notebooks work in VSCode and can be submitted to Kaggle.

### Validate Submission

```bash
python scripts/validate_submission.py submission.csv
```

## Data Format

See Kaggle

## Project Structure

```
├── src/omnilex/           # Core library
│   ├── citations/         # Citation parsing & normalization
│   ├── evaluation/        # Metrics & scoring
│   ├── retrieval/         # Retrieval modules
│   │   ├── ablation/     # Ablation framework
│   │   ├── bm25_index.py # BM25 search
│   │   ├── dense_index.py # BGE-M3 + FAISS
│   │   ├── graph_index.py # Citation graph
│   │   ├── reranker.py   # BGE-reranker
│   │   ├── verifier.py   # LLM verification
│   │   └── fusion.py     # Signal fusion
│   └── llm/               # LLM loading & prompts
├── notebooks/             # Baseline notebooks
├── utils/                 # Data & utility scripts
├── tests/                 # Test suite
└── data/                  # Data directory
```

## Requirements

- Python >= 3.10
- llama-cpp-python (for local LLM inference)
- rank-bm25 (for keyword search)
- faiss-cpu or faiss-gpu (for dense retrieval)
- sentence-transformers, flag-embedding (for BGE-M3 embeddings)
- pandas, numpy, scikit-learn
- networkx, leidenalg (for citation graph)

For Kaggle submissions, you may need to (depending on your solution):

1. Upload your GGUF model as a Kaggle dataset
2. Upload pre-built indices as a Kaggle dataset
3. Package the `omnilex` library

## License

Apache 2.0 - See [LICENSE](LICENSE)

## Contact

For public questions about the competition please use the "Discussion" tab or open an issue on this repository. For private questions reahc out to host on Kaggle or ari.jordan@omnilex.ai
