from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .autograd import MemristorLossPSR
from .coincidence import nfold_channel_count
from .config import SimConfig
from .simulation import run_simulation_sequence_np


def _head_forward_np(
    features: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    loss_type: str,
) -> np.ndarray:
    logits = features @ weight.T + bias
    if loss_type == "mse":
        return logits.squeeze(-1)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def _predict_with_head(
    *,
    theta_np: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    sim_cfg: SimConfig,
    enc_np: np.ndarray,
    n_samples: Optional[int] = None,
    n_swipe: Optional[int] = None,
    swipe_span: Optional[float] = None,
    uq_pass_cfg: bool = False,
) -> np.ndarray:
    if uq_pass_cfg:
        cfg = sim_cfg.replace(
            n_samples=n_samples if n_samples is not None else sim_cfg.n_samples,
            n_swipe=0,
            swipe_span=0.0,
            noise_std=None,
        )
    else:
        cfg = sim_cfg.replace(
            n_samples=n_samples if n_samples is not None else sim_cfg.n_samples,
            n_swipe=n_swipe if n_swipe is not None else sim_cfg.n_swipe,
            swipe_span=swipe_span if swipe_span is not None else sim_cfg.swipe_span,
        )

    feature_cfg = MemristorLossPSR._feature_cfg(cfg)
    preds = run_simulation_sequence_np(
        theta_np,
        enc_np,
        feature_cfg,
        return_class_probs=True,
    )
    if preds.ndim == 1:
        preds = preds[:, None]
    return _head_forward_np(
        preds,
        weight,
        bias,
        loss_type=sim_cfg.loss_type,
    )


@dataclass(frozen=True)
class TrainedPhotonicState:
    """Serializable trained state for inference with the photonic readout head."""

    theta: tuple[float, ...]
    head_weight: tuple[tuple[float, ...], ...]
    head_bias: tuple[float, ...]
    sim_cfg: dict[str, Any]

    @classmethod
    def from_model(cls, model: "PhotonicModel") -> "TrainedPhotonicState":
        weight = model.head.weight.detach().cpu().numpy()
        bias = model.head.bias.detach().cpu().numpy()
        return cls(
            theta=tuple(float(x) for x in model.theta.detach().cpu().numpy()),
            head_weight=tuple(tuple(float(x) for x in row) for row in weight),
            head_bias=tuple(float(x) for x in bias),
            sim_cfg=model.sim_cfg.to_dict(),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainedPhotonicState":
        return cls(
            theta=tuple(float(x) for x in payload["theta"]),
            head_weight=tuple(
                tuple(float(x) for x in row) for row in payload["head_weight"]
            ),
            head_bias=tuple(float(x) for x in payload["head_bias"]),
            sim_cfg=dict(payload["sim_cfg"]),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "TrainedPhotonicState":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "theta": list(self.theta),
            "head_weight": [list(row) for row in self.head_weight],
            "head_bias": list(self.head_bias),
            "sim_cfg": self.sim_cfg,
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    def theta_array(self) -> np.ndarray:
        return np.asarray(self.theta, dtype=np.float64)

    def predict(
        self,
        enc_np: np.ndarray,
        *,
        theta_np: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None,
        n_swipe: Optional[int] = None,
        swipe_span: Optional[float] = None,
        uq_pass_cfg: bool = False,
    ) -> np.ndarray:
        return _predict_with_head(
            theta_np=np.asarray(
                theta_np if theta_np is not None else self.theta,
                dtype=np.float64,
            ),
            weight=np.asarray(self.head_weight, dtype=np.float64),
            bias=np.asarray(self.head_bias, dtype=np.float64),
            sim_cfg=SimConfig.from_dict(self.sim_cfg),
            enc_np=enc_np,
            n_samples=n_samples,
            n_swipe=n_swipe,
            swipe_span=swipe_span,
            uq_pass_cfg=uq_pass_cfg,
        )


class PhotonicModel(torch.nn.Module):
    """
    PyTorch model class for photonic circuit training.
    Args:
        init_theta (Sequence[float]): Initial parameter values.
        enc_np (np.ndarray): Encoded phase values.
        y_np (np.ndarray): Target values.
        phase_idx (Sequence[int]): Indices of phase parameters (trainable mesh phases).
        sim_cfg (SimConfig): Circuit, simulation, and task parameters.
    """

    def __init__(
        self,
        init_theta: Sequence[float],
        enc_np: np.ndarray,
        phase_idx: Sequence[int],
        sim_cfg: SimConfig,
        head: bool = True,
    ) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(init_theta, dtype=torch.float64))
        self.register_buffer("enc", torch.from_numpy(enc_np).double())
        self.phase_idx = phase_idx
        self.sim_cfg = sim_cfg

        if sim_cfg.output_mode == "singles":
            n_features = sim_cfg.n_modes
        else:
            if sim_cfg.working_detectors is None:
                raise ValueError(
                    "coincidence linear head requires working_detectors in SimConfig"
                )
            n_features = nfold_channel_count(
                len(sim_cfg.working_detectors), int(sum(sim_cfg.input_state))
            )
        out_dim = 1 if sim_cfg.loss_type == "mse" else sim_cfg.n_classes
        self.head = (
            torch.nn.Linear(n_features, out_dim, bias=True, dtype=torch.float64)
            if head
            else torch.nn.Identity()
        )

    def export_state(self) -> TrainedPhotonicState:
        if not isinstance(self.head, torch.nn.Linear):
            raise TypeError("PhotonicModel.export_state requires a linear readout head")
        return TrainedPhotonicState.from_model(self)

    def predict(
        self,
        enc_np: np.ndarray,
        *,
        theta_np: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None,
        n_swipe: Optional[int] = None,
        swipe_span: Optional[float] = None,
        uq_pass_cfg: bool = False,
    ) -> np.ndarray:
        """Photonic features plus linear head (no autograd). Matches training readout.

        If ``uq_pass_cfg`` is True, use discrete UQ settings (``n_swipe=0``, ``swipe_span=0``,
        ``noise_std=None``) like :meth:`Experiment.run_uncertainty_analysis`, while still
        honoring ``n_samples`` when provided.
        """
        if not isinstance(self.head, torch.nn.Linear):
            raise TypeError("PhotonicModel.predict requires a linear readout head")
        return _predict_with_head(
            theta_np=np.asarray(
                theta_np if theta_np is not None else self.theta.detach().cpu().numpy(),
                dtype=np.float64,
            ),
            weight=self.head.weight.detach().cpu().numpy(),
            bias=self.head.bias.detach().cpu().numpy(),
            sim_cfg=self.sim_cfg,
            enc_np=enc_np,
            n_samples=n_samples,
            n_swipe=n_swipe,
            swipe_span=swipe_span,
            uq_pass_cfg=uq_pass_cfg,
        )

    def forward(
        self, y: Tensor, n_samples: int, n_swipe: int, swipe_span: float
    ) -> Tensor:
        """
        Computes the loss from photonic features with a trainable linear head.
        Args:
            y (Tensor): Regression targets or class labels.
            n_samples (int): Number of samples for the Sampler.
            n_swipe (int): Number of phase points per data point (0 for discrete).
            swipe_span (float): Total phase span for swiping.
        Returns:
            Tensor: Scalar loss value.
        """
        cfg = self.sim_cfg.replace(
            n_samples=n_samples, n_swipe=n_swipe, swipe_span=swipe_span
        )
        features = MemristorLossPSR.apply(
            self.theta,
            self.enc,
            self.phase_idx,
            cfg,
        )
        preds = self.head(features)
        if self.sim_cfg.loss_type == "mse":
            return F.mse_loss(preds.squeeze(-1), y.double())
        return F.cross_entropy(preds, y.long())
