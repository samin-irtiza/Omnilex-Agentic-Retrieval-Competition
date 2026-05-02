"""YAML experiment configuration parser for ablation framework."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


EXPERIMENT_PRESETS = {
    "exp_baseline": {
        "name": "exp_baseline",
        "description": "BM25-only baseline",
        "components": {
            "bm25": True,
            "dense": False,
            "graph": False,
            "rrf_fusion": False,
            "reranker": False,
            "verifier": False,
        },
        "laws_corpus_path": None,
        "courts_corpus_path": None,
        "index_cache_dir": None,
    },
    "exp_dense_only": {
        "name": "exp_dense_only",
        "description": "Dense retrieval only (BGE-M3 + FAISS)",
        "components": {
            "bm25": False,
            "dense": True,
            "graph": False,
            "rrf_fusion": False,
            "reranker": False,
            "verifier": False,
        },
        "laws_corpus_path": None,
        "courts_corpus_path": None,
        "index_cache_dir": None,
    },
    "exp_bm25_dense": {
        "name": "exp_bm25_dense",
        "description": "BM25 + Dense retrieval with RRF",
        "components": {
            "bm25": True,
            "dense": True,
            "graph": False,
            "rrf_fusion": True,
            "reranker": False,
            "verifier": False,
        },
        "laws_corpus_path": None,
        "courts_corpus_path": None,
        "index_cache_dir": None,
    },
    "exp_full_retrieval": {
        "name": "exp_full_retrieval",
        "description": "BM25 + Dense + Graph retrieval (no fusion)",
        "components": {
            "bm25": True,
            "dense": True,
            "graph": True,
            "rrf_fusion": False,
            "reranker": False,
            "verifier": False,
        },
        "laws_corpus_path": None,
        "courts_corpus_path": None,
        "index_cache_dir": None,
    },
    "exp_full_rrf": {
        "name": "exp_full_rrf",
        "description": "Full retrieval + RRF fusion",
        "components": {
            "bm25": True,
            "dense": True,
            "graph": True,
            "rrf_fusion": True,
            "reranker": False,
            "verifier": False,
        },
        "laws_corpus_path": None,
        "courts_corpus_path": None,
        "index_cache_dir": None,
    },
    "exp_full_reranker": {
        "name": "exp_full_reranker",
        "description": "Full retrieval + RRF + Reranker",
        "components": {
            "bm25": True,
            "dense": True,
            "graph": True,
            "rrf_fusion": True,
            "reranker": True,
            "verifier": False,
        },
        "laws_corpus_path": None,
        "courts_corpus_path": None,
        "index_cache_dir": None,
    },
    "exp_full_pipeline": {
        "name": "exp_full_pipeline",
        "description": "Full pipeline with all components",
        "components": {
            "bm25": True,
            "dense": True,
            "graph": True,
            "rrf_fusion": True,
            "reranker": True,
            "verifier": True,
        },
        "laws_corpus_path": None,
        "courts_corpus_path": None,
        "index_cache_dir": None,
    },
}


class ExperimentConfig:
    """Experiment configuration for ablation studies."""

    def __init__(
        self,
        name: str,
        description: str = "",
        components: dict | None = None,
        signal_weights: dict | None = None,
        dense_index_preset: str = "balanced",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        verifier_model: str = "qwen2.5-7b-q4_k_m.gguf",
        top_k: int = 50,
        fusion_top_k: int = 20,
        reranker_top_k: int = 10,
        verifier_threshold: float = 0.5,
        laws_corpus_path: str | Path | None = None,
        courts_corpus_path: str | Path | None = None,
        index_cache_dir: str | Path | None = None,
    ):
        """Initialize experiment config.

        Args:
            name: Experiment name
            description: Description of the experiment
            components: Dict of component toggles
            signal_weights: Weights for RRF fusion
            dense_index_preset: FAISS index preset
            reranker_model: Reranker model name
            verifier_model: Verifier model path
            top_k: Initial retrieval top_k
            fusion_top_k: After fusion top_k
            reranker_top_k: After reranking top_k
            verifier_threshold: Verifier score threshold
            laws_corpus_path: Path to laws corpus CSV file
            courts_corpus_path: Path to courts corpus CSV file
            index_cache_dir: Directory for caching built indices
        """
        self.name = name
        self.description = description
        self.components = components or {}
        self.signal_weights = signal_weights or {}
        self.dense_index_preset = dense_index_preset
        self.reranker_model = reranker_model
        self.verifier_model = verifier_model
        self.top_k = top_k
        self.fusion_top_k = fusion_top_k
        self.reranker_top_k = reranker_top_k
        self.verifier_threshold = verifier_threshold
        self._laws_corpus_path = Path(laws_corpus_path) if laws_corpus_path else None
        self._courts_corpus_path = Path(courts_corpus_path) if courts_corpus_path else None
        self._index_cache_dir = Path(index_cache_dir) if index_cache_dir else None

    @property
    def laws_corpus_path(self) -> Path | None:
        """Get laws corpus path."""
        return self._laws_corpus_path

    @laws_corpus_path.setter
    def laws_corpus_path(self, value):
        """Set laws corpus path, converting string to Path if needed."""
        self._laws_corpus_path = Path(value) if value else None

    @property
    def courts_corpus_path(self) -> Path | None:
        """Get courts corpus path."""
        return self._courts_corpus_path

    @courts_corpus_path.setter
    def courts_corpus_path(self, value):
        """Set courts corpus path, converting string to Path if needed."""
        self._courts_corpus_path = Path(value) if value else None

    @property
    def index_cache_dir(self) -> Path | None:
        """Get index cache directory."""
        return self._index_cache_dir

    @index_cache_dir.setter
    def index_cache_dir(self, value):
        """Set index cache directory, converting string to Path if needed."""
        self._index_cache_dir = Path(value) if value else None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            ExperimentConfig instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            config_dict = yaml.safe_load(f)

        # Extract fields with defaults matching __init__ parameters
        laws_path = config_dict.get("laws_corpus_path")
        courts_path = config_dict.get("courts_corpus_path")
        cache_dir = config_dict.get("index_cache_dir")

        instance = cls(
            name=config_dict.get("name", ""),
            description=config_dict.get("description", ""),
            components=config_dict.get("components", {}),
            signal_weights=config_dict.get("signal_weights", {}),
            dense_index_preset=config_dict.get("dense_index_preset", "balanced"),
            reranker_model=config_dict.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
            verifier_model=config_dict.get("verifier_model", "qwen2.5-7b-q4_k_m.gguf"),
            top_k=config_dict.get("top_k", 50),
            fusion_top_k=config_dict.get("fusion_top_k", 20),
            reranker_top_k=config_dict.get("reranker_top_k", 10),
            verifier_threshold=config_dict.get("verifier_threshold", 0.5),
            laws_corpus_path=Path(laws_path) if laws_path else None,
            courts_corpus_path=Path(courts_path) if courts_path else None,
            index_cache_dir=Path(cache_dir) if cache_dir else None,
        )

        errors = instance.validate()
        if errors:
            logger.warning(f"Config validation errors: {errors}")

        return instance

    @classmethod
    def from_preset(cls, preset_name: str) -> ExperimentConfig:
        """Load configuration from preset.

        Args:
            preset_name: Name of preset (e.g., "exp_baseline")

        Returns:
            ExperimentConfig instance

        Raises:
            ValueError: If preset not found
        """
        if preset_name not in EXPERIMENT_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {list(EXPERIMENT_PRESETS.keys())}"
            )

        preset = EXPERIMENT_PRESETS[preset_name]
        
        # Get paths and convert to Path objects if they exist
        laws_path = preset.get("laws_corpus_path")
        courts_path = preset.get("courts_corpus_path")
        cache_dir = preset.get("index_cache_dir")
        
        return cls(
            name=preset["name"],
            description=preset["description"],
            components=preset["components"],
            laws_corpus_path=Path(laws_path) if laws_path else None,
            courts_corpus_path=Path(courts_path) if courts_path else None,
            index_cache_dir=Path(cache_dir) if cache_dir else None,
        )

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not self.name:
            errors.append("Experiment name is required")

        valid_components = ["bm25", "dense", "graph", "rrf_fusion", "reranker", "verifier"]
        for comp in self.components:
            if comp not in valid_components:
                errors.append(f"Unknown component: {comp}")

        if self.verifier_threshold < 0 or self.verifier_threshold > 1:
            errors.append("verifier_threshold must be between 0 and 1")

        # Warn if BM25 is enabled but corpus paths are not configured
        if self.components.get("bm25"):
            if not self.laws_corpus_path:
                logger.warning(
                    "BM25 is enabled but laws_corpus_path is not configured. "
                    "Set laws_corpus_path in config or preset."
                )
            if not self.courts_corpus_path:
                logger.warning(
                    "BM25 is enabled but courts_corpus_path is not configured. "
                    "Set courts_corpus_path in config or preset."
                )
            if not self.index_cache_dir:
                logger.warning(
                    "index_cache_dir is not configured. "
                    "BM25 indices will be built from scratch each run."
                )

        return errors

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Dict representation of config
        """
        return {
            "name": self.name,
            "description": self.description,
            "components": self.components,
            "signal_weights": self.signal_weights,
            "dense_index_preset": self.dense_index_preset,
            "reranker_model": self.reranker_model,
            "verifier_model": self.verifier_model,
            "top_k": self.top_k,
            "fusion_top_k": self.fusion_top_k,
            "reranker_top_k": self.reranker_top_k,
            "verifier_threshold": self.verifier_threshold,
            "laws_corpus_path": str(self.laws_corpus_path) if self.laws_corpus_path else None,
            "courts_corpus_path": str(self.courts_corpus_path) if self.courts_corpus_path else None,
            "index_cache_dir": str(self.index_cache_dir) if self.index_cache_dir else None,
        }
