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
    },
    "exp_full_retrieval": {
        "name": "exp_full_retrieval",
        "description": "BM25 + Dense + Graph with RRF",
        "components": {
            "bm25": True,
            "dense": True,
            "graph": True,
            "rrf_fusion": True,
            "reranker": False,
            "verifier": False,
        },
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
        return cls(
            name=preset["name"],
            description=preset["description"],
            components=preset["components"],
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
        }
