from __future__ import annotations

import copy
import logging
from collections import OrderedDict

import torch
import torch.nn as nn


class EarlyStopping:
	"""Stateful early stopping utility with optional best-weight restoration."""

	def __init__(
		self,
		patience: int,
		min_delta: float,
		mode: str,
		restore_best_weights: bool,
	) -> None:
		if patience < 0:
			raise ValueError("patience must be >= 0")
		if min_delta < 0:
			raise ValueError("min_delta must be >= 0")
		if mode not in {"min", "max"}:
			raise ValueError("mode must be one of: 'min', 'max'")

		self.patience = patience
		self.min_delta = float(min_delta)
		self.mode = mode
		self.restore_best_weights = restore_best_weights

		self.best_score: float | None = None
		self.best_epoch: int | None = None
		self.best_state_dict: OrderedDict[str, torch.Tensor] | None = None
		self.counter = 0
		self.step_count = 0
		self._logger = logging.getLogger(__name__)

	def _is_improvement(self, metric: float) -> bool:
		if self.best_score is None:
			return True
		if self.mode == "min":
			return metric < (self.best_score - self.min_delta)
		return metric > (self.best_score + self.min_delta)

	@staticmethod
	def _clone_state_dict(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
		state = model.state_dict()
		cloned: OrderedDict[str, torch.Tensor] = OrderedDict()
		for key, value in state.items():
			if torch.is_tensor(value):
				cloned[key] = value.detach().cpu().clone()
			else:
				cloned[key] = copy.deepcopy(value)
		return cloned

	def step(self, metric: float, model: nn.Module) -> bool:
		"""Update early stopping state and return True when training should stop."""
		self.step_count += 1
		metric_value = float(metric)

		if self._is_improvement(metric_value):
			self.best_score = metric_value
			self.best_epoch = self.step_count
			self.best_state_dict = self._clone_state_dict(model)
			self.counter = 0
			self._logger.info(
				"EarlyStopping improvement: epoch=%s, best_score=%.6f",
				self.best_epoch,
				self.best_score,
			)
			return False

		self.counter += 1
		should_stop = self.counter >= self.patience
		self._logger.info(
			"EarlyStopping no improvement: epoch=%s, metric=%.6f, best_score=%.6f, counter=%s/%s",
			self.step_count,
			metric_value,
			self.best_score if self.best_score is not None else float("nan"),
			self.counter,
			self.patience,
		)
		if should_stop:
			self._logger.info(
				"EarlyStopping triggered at epoch=%s. Best epoch=%s with best_score=%.6f",
				self.step_count,
				self.best_epoch,
				self.best_score if self.best_score is not None else float("nan"),
			)
		return should_stop

	def restore(self, model: nn.Module) -> bool:
		"""Restore best model weights if available and enabled."""
		if not self.restore_best_weights:
			self._logger.info("EarlyStopping restore skipped: restore_best_weights=False")
			return False
		if self.best_state_dict is None:
			self._logger.warning("EarlyStopping restore skipped: no best_state_dict saved")
			return False

		model.load_state_dict(self.best_state_dict)
		self._logger.info(
			"Restored best weights from epoch=%s with best_score=%.6f",
			self.best_epoch,
			self.best_score if self.best_score is not None else float("nan"),
		)
		return True


__all__ = ["EarlyStopping"]


# Example usage and validation
if __name__ == "__main__":
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
	)

	torch.manual_seed(42)
	model = nn.Linear(4, 2)
	stopper = EarlyStopping(
		patience=2,
		min_delta=0.01,
		mode="min",
		restore_best_weights=True,
	)

	metrics = [1.0, 0.8, 0.79, 0.791, 0.792]
	stop_epoch: int | None = None
	best_output_reference: torch.Tensor | None = None
	probe = torch.randn(2, 4)

	for epoch, metric in enumerate(metrics, start=1):
		with torch.no_grad():
			for param in model.parameters():
				param.add_(0.05 * torch.randn_like(param))

		previous_best_epoch = stopper.best_epoch
		if stopper.step(metric=metric, model=model):
			stop_epoch = epoch
			break

		if stopper.best_epoch == epoch and stopper.best_epoch != previous_best_epoch:
			best_output_reference = model(probe).detach().clone()

	assert stop_epoch == 4, f"Expected stop at epoch 4, got: {stop_epoch}"
	assert stopper.best_epoch == 2, f"Expected best epoch 2, got: {stopper.best_epoch}"
	assert stopper.best_state_dict is not None, "best_state_dict should be saved"

	with torch.no_grad():
		for param in model.parameters():
			param.zero_()

	restored = stopper.restore(model)
	assert restored, "Expected successful restore"

	restored_output = model(probe).detach()
	assert best_output_reference is not None, "Missing best output reference"
	assert torch.allclose(restored_output, best_output_reference, atol=1e-6), "Restore did not recover best weights"

	print(
		"EarlyStopping check passed. "
		f"best_epoch={stopper.best_epoch}, best_score={stopper.best_score}, stop_epoch={stop_epoch}"
	)
