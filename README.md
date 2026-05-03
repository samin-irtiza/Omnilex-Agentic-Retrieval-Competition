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

### Google Colab Setup

To run notebooks in Google Colab with persistent caching (avoids 15-25 min setup delays):

1. **One-Time Setup:**
   - Create cache directory in Google Drive: `MyDrive/omnilex-cache/`
   - Populate caches (uv wheels, Kaggle data, pre-built indices) - see [notebooks/colab_startup.ipynb](notebooks/colab_startup.ipynb) for instructions
   - **New:** The uv cache now uses a **tar.gz archive** (`uv-cache.tar.gz`) for faster sync (~2-3 min vs ~10 min with old approach)
   - After first run, execute the "Manual Archive Update" cell to create the archive on Drive

2. **Per Session:**
   - Open [notebooks/colab_startup.ipynb](notebooks/colab_startup.ipynb) in Colab
   - Run all cells (mounts Drive, extracts uv cache from archive, creates symlinks)
   - Open desired baseline notebook and run as usual

**Benefits:**
- Reduces uv cache sync from ~10 minutes to ~2-3 minutes using tar.gz archive
- Auto-updates archive when new packages are installed (checksum-based detection)
- Caches packages, data, and indices on Google Drive
- Falls back to PyPI downloads if archive is missing/corrupted

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
