from __future__ import annotations

from collections import Counter
from typing import Any

from ..logging_config import get_logger

logger = get_logger(__name__)


class SimulationLogger:
    """Collects runtime statistics for photonic simulations."""

    def __init__(self):
        self.call_count = 0
        self.total_time = 0.0
        self.samples_counter = Counter()
        self.circuit_call_count = 0
        self.circuit_total_time = 0.0

    def log(self, elapsed: float, n_samples: int) -> None:
        self.call_count += 1
        self.total_time += elapsed
        self.samples_counter[n_samples] += 1

    def log_circuit(self, elapsed: float) -> None:
        self.circuit_call_count += 1
        self.circuit_total_time += elapsed

    def log_circuits(self, elapsed: float, count: int = 1) -> None:
        self.circuit_call_count += count
        self.circuit_total_time += elapsed

    def report(self) -> None:
        lines = [
            f"Circuit sequence runs: {self.call_count}",
            f"Total sequence time: {self.total_time:.3f}s",
        ]
        if self.call_count > 0:
            lines.append(
                f"Avg time per sequence: {self.total_time / self.call_count:.6f}s"
            )
        sample_parts = [f"{n}×{freq}" for n, freq in self.samples_counter.items()]
        lines.append(f"Sampler sample counts: {', '.join(sample_parts) or 'none'}")
        lines.append(f"Individual circuit sims: {self.circuit_call_count}")
        lines.append(f"Total circuit sim time: {self.circuit_total_time:.3f}s")
        if self.circuit_call_count > 0:
            lines.append(
                f"Avg time per circuit sim: "
                f"{self.circuit_total_time / self.circuit_call_count:.6f}s"
            )
        logger.info("Simulation statistics:\n  " + "\n  ".join(lines))

    def stats_dict(self) -> dict[str, Any]:
        return {
            "call_count": self.call_count,
            "total_time": self.total_time,
            "samples_counter": dict(self.samples_counter),
            "circuit_call_count": self.circuit_call_count,
            "circuit_total_time": self.circuit_total_time,
            "avg_time_per_sequence": self.total_time / self.call_count
            if self.call_count
            else 0,
            "avg_time_per_circuit": (
                self.circuit_total_time / self.circuit_call_count
                if self.circuit_call_count
                else 0
            ),
        }


sim_logger = SimulationLogger()
