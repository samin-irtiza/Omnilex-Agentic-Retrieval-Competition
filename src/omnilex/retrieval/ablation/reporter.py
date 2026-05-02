"""Results aggregation and comparison table for ablation framework.

Generates comparison reports and exports to CSV/JSON.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ResultsReporter:
    """Aggregates and reports experiment results."""

    def __init__(self, output_dir: str | Path = "experiments"):
        """Initialize results reporter.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []

    def add_result(
        self,
        experiment_name: str,
        config: dict,
        metrics: dict,
        per_query_f1: list[float] | None = None,
    ) -> None:
        """Add experiment result.

        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration
            metrics: Aggregate metrics dict
            per_query_f1: Per-query F1 scores
        """
        self.results.append(
            {
                "experiment_name": experiment_name,
                "config": config,
                "metrics": metrics,
                "per_query_f1": per_query_f1 or [],
            }
        )

    def generate_comparison_table(self) -> str:
        """Generate comparison table as string.

        Returns:
            Formatted comparison table
        """
        if not self.results:
            return "No results to compare."

        # Collect all metric keys
        metric_keys = set()
        for res in self.results:
            metric_keys.update(res["metrics"].keys())
        metric_keys = sorted(metric_keys)

        # Build table rows
        header = ["Experiment"] + list(metric_keys)
        rows = []

        for res in self.results:
            row = [res["experiment_name"]]
            for mk in metric_keys:
                val = res["metrics"].get(mk, "N/A")
                row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            rows.append(row)

        # Calculate column widths
        col_widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]

        # Format table
        lines = []
        header_line = " | ".join(f"{header[i]:<{col_widths[i]}}" for i in range(len(header)))
        lines.append(header_line)

        separator = "-+-".join("-" * col_widths[i] for i in range(len(header)))
        lines.append(separator)

        for row in rows:
            line = " | ".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row)))
            lines.append(line)

        return "\n".join(lines)

    def export_csv(self, filename: str = "results.csv") -> Path:
        """Export results to CSV.

        Args:
            filename: Output filename

        Returns:
            Path to saved file
        """
        path = self.output_dir / filename

        if not self.results:
            return path

        # Collect all metric keys
        metric_keys = set()
        for res in self.results:
            metric_keys.update(res["metrics"].keys())
        metric_keys = sorted(metric_keys)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["experiment_name"] + list(metric_keys))

            for res in self.results:
                row = [res["experiment_name"]]
                for mk in metric_keys:
                    row.append(res["metrics"].get(mk, ""))
                writer.writerow(row)

        return path

    def export_json(self, filename: str = "results.json") -> Path:
        """Export results to JSON.

        Args:
            filename: Output filename

        Returns:
            Path to saved file
        """
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        return path

    def print_summary(self) -> None:
        """Print summary of all results."""
        table = self.generate_comparison_table()
        print(table)
