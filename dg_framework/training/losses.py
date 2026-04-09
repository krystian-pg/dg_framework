from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_1d_float_tensor(values: Sequence[float] | torch.Tensor, name: str) -> torch.Tensor:
	tensor = values.detach().clone().float() if torch.is_tensor(values) else torch.tensor(values, dtype=torch.float32)
	if tensor.ndim != 1:
		raise ValueError(f"{name} must be a 1D sequence or tensor")
	if tensor.numel() == 0:
		raise ValueError(f"{name} cannot be empty")
	return tensor


def _inverse_frequency_weights(
	class_frequencies: Sequence[float] | torch.Tensor,
	eps: float = 1e-12,
	normalize: bool = True,
) -> torch.Tensor:
	freq = _to_1d_float_tensor(class_frequencies, "class_frequencies")
	if torch.any(freq < 0):
		raise ValueError("class_frequencies must contain non-negative values")
	total = float(freq.sum().item())
	if total <= 0:
		raise ValueError("class_frequencies must sum to a positive value")

	weights = 1.0 / (freq + eps)
	if normalize:
		weights = weights / weights.mean().clamp(min=eps)
	return weights


class FocalLoss(nn.Module):
	"""Multi-class focal loss based on cross entropy."""

	def __init__(
		self,
		gamma: float = 2.0,
		alpha: float | Sequence[float] | torch.Tensor | None = None,
		reduction: str = "mean",
		label_smoothing: float = 0.0,
	) -> None:
		super().__init__()
		if gamma < 0:
			raise ValueError("gamma must be >= 0")
		if reduction not in {"none", "mean", "sum"}:
			raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")
		if not 0.0 <= label_smoothing < 1.0:
			raise ValueError("label_smoothing must be in [0, 1)")

		self.gamma = float(gamma)
		self.reduction = reduction
		self.label_smoothing = float(label_smoothing)
		self.alpha_scalar: float | None = None
		self.alpha_vector: torch.Tensor | None = None

		if alpha is None:
			return
		if isinstance(alpha, (float, int)):
			if alpha < 0:
				raise ValueError("alpha must be >= 0")
			self.alpha_scalar = float(alpha)
			return
		self.alpha_vector = _to_1d_float_tensor(alpha, "alpha")

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		ce = F.cross_entropy(
			logits,
			targets,
			reduction="none",
			label_smoothing=self.label_smoothing,
		)
		pt = torch.exp(-ce)
		loss = ((1.0 - pt) ** self.gamma) * ce

		if self.alpha_scalar is not None:
			loss = self.alpha_scalar * loss
		elif self.alpha_vector is not None:
			alpha = self.alpha_vector.to(device=logits.device, dtype=logits.dtype)
			if alpha.numel() != logits.size(1):
				raise ValueError(
					f"alpha length ({alpha.numel()}) must match number of classes ({logits.size(1)})"
				)
			loss = alpha[targets] * loss

		if self.reduction == "none":
			return loss
		if self.reduction == "sum":
			return loss.sum()
		return loss.mean()


def build_cross_entropy(
	label_smoothing: float = 0.0,
	class_weights: Sequence[float] | torch.Tensor | None = None,
) -> nn.Module:
	if not 0.0 <= label_smoothing < 1.0:
		raise ValueError("label_smoothing must be in [0, 1)")
	weight_tensor = None if class_weights is None else _to_1d_float_tensor(class_weights, "class_weights")
	return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)


def build_focal_loss(
	gamma: float = 2.0,
	alpha: float | Sequence[float] | torch.Tensor | None = None,
	reduction: str = "mean",
	label_smoothing: float = 0.0,
) -> nn.Module:
	return FocalLoss(
		gamma=gamma,
		alpha=alpha,
		reduction=reduction,
		label_smoothing=label_smoothing,
	)


def build_class_weighted_ce(
	class_frequencies: Sequence[float] | torch.Tensor | None = None,
	class_counts: Sequence[float] | torch.Tensor | None = None,
	class_weights: Sequence[float] | torch.Tensor | None = None,
	label_smoothing: float = 0.0,
	normalize: bool = True,
) -> nn.Module:
	if class_weights is not None:
		weights = _to_1d_float_tensor(class_weights, "class_weights")
	elif class_frequencies is not None:
		weights = _inverse_frequency_weights(class_frequencies, normalize=normalize)
	elif class_counts is not None:
		counts = _to_1d_float_tensor(class_counts, "class_counts")
		if torch.any(counts < 0):
			raise ValueError("class_counts must contain non-negative values")
		total = float(counts.sum().item())
		if total <= 0:
			raise ValueError("class_counts must sum to a positive value")
		frequencies = counts / total
		weights = _inverse_frequency_weights(frequencies, normalize=normalize)
	else:
		raise ValueError(
			"class_weighted_ce requires one of: class_weights, class_frequencies, class_counts"
		)

	return build_cross_entropy(label_smoothing=label_smoothing, class_weights=weights)


LOSSES: dict[str, Callable[..., nn.Module]] = {
	"cross_entropy": build_cross_entropy,
	"focal_loss": build_focal_loss,
	"class_weighted_ce": build_class_weighted_ce,
}


def get_loss(name: str, **kwargs) -> nn.Module:
	"""Create a loss module from the central registry."""
	if name not in LOSSES:
		available = ", ".join(sorted(LOSSES))
		raise KeyError(f"Unknown loss '{name}'. Available losses: {available}")
	return LOSSES[name](**kwargs)


__all__ = ["LOSSES", "FocalLoss", "get_loss"]


# Example usage and validation
if __name__ == "__main__":
	torch.manual_seed(42)

	batch_size = 8
	num_classes = 7
	logits = torch.randn(batch_size, num_classes)
	targets = torch.randint(0, num_classes, (batch_size,))

	ce_loss = get_loss("cross_entropy", label_smoothing=0.1)
	focal_loss = get_loss("focal_loss", gamma=2.0, alpha=0.5, label_smoothing=0.05)
	weighted_ce_loss = get_loss(
		"class_weighted_ce",
		class_counts=[120, 90, 60, 30, 20, 10, 5],
		label_smoothing=0.1,
	)

	ce_value = ce_loss(logits, targets)
	focal_value = focal_loss(logits, targets)
	weighted_ce_value = weighted_ce_loss(logits, targets)

	assert isinstance(ce_loss, nn.Module), "cross_entropy must return nn.Module"
	assert isinstance(focal_loss, nn.Module), "focal_loss must return nn.Module"
	assert isinstance(weighted_ce_loss, nn.Module), "class_weighted_ce must return nn.Module"
	assert ce_value.ndim == 0, "cross_entropy output must be scalar"
	assert focal_value.ndim == 0, "focal_loss output must be scalar"
	assert weighted_ce_value.ndim == 0, "class_weighted_ce output must be scalar"

	weights = weighted_ce_loss.weight
	assert weights is not None, "class_weighted_ce should define class weights"
	assert weights[0] < weights[-1], "Rarer classes should get larger weights"

	print(
		"Loss registry check passed. "
		f"available={sorted(LOSSES.keys())}, "
		f"ce={ce_value.item():.4f}, focal={focal_value.item():.4f}, weighted_ce={weighted_ce_value.item():.4f}"
	)
