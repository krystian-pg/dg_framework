from __future__ import annotations

import logging
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
	from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
	tqdm = None

try:
	from dg_framework.config import CFG, Config
	from dg_framework.utils.experiment import setup_experiment
except ModuleNotFoundError:
	project_root = Path(__file__).resolve().parents[1]
	if str(project_root) not in sys.path:
		sys.path.append(str(project_root))
	from config import CFG, Config
	from utils.experiment import setup_experiment


class Trainer:
	"""End-to-end training loop with AMP, clipping, checkpointing, and history tracking."""

	def __init__(
		self,
		model: nn.Module,
		loaders: dict[str, Any] | tuple[Any, Any],
		loss_fn: nn.Module,
		optimiser: torch.optim.Optimizer,
		scheduler: Any,
		early_stopping: Any,
		config: Config,
	) -> None:
		self.model = model
		self.loss_fn = loss_fn
		self.optimiser = optimiser
		self.scheduler = scheduler
		self.early_stopping = early_stopping
		self.config = config
		self.logger = logging.getLogger(__name__)

		self.train_loader, self.val_loader = self._resolve_loaders(loaders)
		self.device = next(self.model.parameters()).device

		self.use_amp = bool(self.config.train.amp and self.device.type == "cuda")
		scaler_device = "cuda" if self.device.type == "cuda" else "cpu"
		self.grad_scaler = torch.amp.GradScaler(device=scaler_device, enabled=self.use_amp)
		self.grad_clip = self.config.train.grad_clip
		self.grad_accum_steps = max(1, int(self.config.train.grad_accum_steps))
		self.show_progress = bool(getattr(self.config.train, "show_progress", True))
		self.progress_log_interval = max(1, int(getattr(self.config.train, "progress_log_interval", 20)))
		self._use_tqdm = bool(self.show_progress and tqdm is not None and sys.stderr.isatty())
		if self.show_progress and tqdm is None:
			self.logger.warning(
				"train.show_progress=True but tqdm is not installed; progress bars are disabled."
			)
		elif self.show_progress and not sys.stderr.isatty():
			self.logger.warning(
				"Progress bar disabled because stderr is not a TTY. Falling back to periodic step logs."
			)

		self.experiment_path = setup_experiment(self.config)
		self.best_checkpoint_path = self.experiment_path / "checkpoints" / "best.pt"
		self.best_score: float | None = None

		self._monitor_key, self._monitor_mode = self._resolve_monitor()

	def _resolve_loaders(self, loaders: dict[str, Any] | tuple[Any, Any]) -> tuple[Any, Any]:
		if isinstance(loaders, dict):
			if "train" not in loaders or "val" not in loaders:
				raise KeyError("loaders dict must provide 'train' and 'val'")
			return loaders["train"], loaders["val"]

		if isinstance(loaders, tuple) and len(loaders) >= 2:
			return loaders[0], loaders[1]

		raise TypeError("loaders must be dict with train/val or tuple(train_loader, val_loader, ...)")

	def _resolve_monitor(self) -> tuple[str, str]:
		metrics = set(getattr(self.config.evaluation, "metrics", []))
		if "accuracy" in metrics:
			return "val_accuracy", "max"
		return "val_loss", "min"

	@staticmethod
	def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
		if not isinstance(batch, (tuple, list)) or len(batch) < 2:
			raise ValueError("Expected batch as (inputs, labels, ...) tuple")
		return batch[0], batch[1]

	def _compute_ece(self, logits: torch.Tensor, labels: torch.Tensor, n_bins: int) -> float:
		probs = torch.softmax(logits, dim=1)
		confidences, predictions = probs.max(dim=1)
		accuracies = predictions.eq(labels)

		ece = torch.zeros(1, device=logits.device)
		bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)

		for idx in range(n_bins):
			low = bin_boundaries[idx]
			high = bin_boundaries[idx + 1]
			in_bin = confidences.gt(low) & confidences.le(high)
			frac = in_bin.float().mean()
			if frac.item() > 0:
				acc_bin = accuracies[in_bin].float().mean()
				conf_bin = confidences[in_bin].mean()
				ece += torch.abs(acc_bin - conf_bin) * frac

		return float(ece.item())

	def _compute_macro_f1(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
		preds = logits.argmax(dim=1)
		num_classes = logits.size(1)
		f1_scores: list[float] = []

		for class_idx in range(num_classes):
			tp = ((preds == class_idx) & (labels == class_idx)).sum().float()
			fp = ((preds == class_idx) & (labels != class_idx)).sum().float()
			fn = ((preds != class_idx) & (labels == class_idx)).sum().float()

			precision = tp / (tp + fp + 1e-12)
			recall = tp / (tp + fn + 1e-12)
			f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
			f1_scores.append(float(f1.item()))

		return sum(f1_scores) / max(len(f1_scores), 1)

	def _compute_topk_accuracy(self, logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
		k_eff = min(k, logits.size(1))
		_, topk_indices = logits.topk(k_eff, dim=1, largest=True, sorted=True)
		correct = topk_indices.eq(labels.unsqueeze(1)).any(dim=1)
		return float(correct.float().mean().item())

	def _compute_eval_metrics(
		self,
		loss_value: float,
		epoch_logits: torch.Tensor,
		epoch_labels: torch.Tensor,
	) -> dict[str, float]:
		metrics = set(getattr(self.config.evaluation, "metrics", []))
		results: dict[str, float] = {"loss": float(loss_value)}

		if "accuracy" in metrics:
			acc = epoch_logits.argmax(dim=1).eq(epoch_labels).float().mean()
			results["accuracy"] = float(acc.item())

		if "f1" in metrics:
			results["f1"] = self._compute_macro_f1(epoch_logits, epoch_labels)

		if "ece" in metrics:
			n_bins = int(getattr(self.config.evaluation, "ece_bins", 15))
			results["ece"] = self._compute_ece(epoch_logits, epoch_labels, n_bins=n_bins)

		for k in tuple(getattr(self.config.evaluation, "top_k", ())):
			if isinstance(k, int) and k > 0:
				results[f"top{k}"] = self._compute_topk_accuracy(epoch_logits, epoch_labels, k)

		return results

	def _run_external_evaluator(
		self,
		epoch_logits: torch.Tensor,
		epoch_labels: torch.Tensor,
		fallback_metrics: dict[str, float],
	) -> dict[str, float]:
		evaluator: Callable[..., dict[str, float]] | None = getattr(self, "evaluator", None)
		if evaluator is None:
			return fallback_metrics

		result = evaluator(logits=epoch_logits, labels=epoch_labels)
		if not isinstance(result, dict):
			raise TypeError("evaluator must return dict[str, float]")

		merged = dict(fallback_metrics)
		for key, value in result.items():
			merged[str(key)] = float(value)
		return merged

	def _step_scheduler(self, val_loss: float) -> None:
		if self.scheduler is None:
			return

		scheduler_type = getattr(self.config.train.scheduler, "scheduler_type", "none")
		if scheduler_type == "plateau":
			self.scheduler.step(val_loss)
		else:
			self.scheduler.step()

	def _is_improvement(self, score: float) -> bool:
		if self.best_score is None:
			return True
		if self._monitor_mode == "max":
			return score > self.best_score
		return score < self.best_score

	def _save_best_checkpoint(self, epoch: int, val_metrics: dict[str, float]) -> None:
		torch.save(
			{
				"epoch": epoch,
				"model_state_dict": self.model.state_dict(),
				"optimiser_state_dict": self.optimiser.state_dict(),
				"scheduler_state_dict": None if self.scheduler is None else self.scheduler.state_dict(),
				"scaler_state_dict": self.grad_scaler.state_dict(),
				"best_score": self.best_score,
				"monitor_key": self._monitor_key,
				"val_metrics": val_metrics,
				"config": self.config,
			},
			self.best_checkpoint_path,
		)

	def train_epoch(self) -> dict[str, float]:
		self.model.train()
		total_loss = 0.0
		total_samples = 0
		use_tqdm = self._use_tqdm
		num_steps = len(self.train_loader)

		self.optimiser.zero_grad(set_to_none=True)

		pbar = None
		if use_tqdm:
			pbar = tqdm(
				total=num_steps,
				desc="train",
				leave=True,
				dynamic_ncols=True,
				mininterval=0.2,
				file=sys.stderr,
			)

		for step_idx, batch in enumerate(self.train_loader, start=1):
			inputs, labels = self._unpack_batch(batch)
			inputs = inputs.to(self.device, non_blocking=True)
			labels = labels.to(self.device, non_blocking=True)
			batch_size = labels.size(0)

			with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
				logits = self.model(inputs)
				loss = self.loss_fn(logits, labels)

			raw_loss = loss.detach()
			loss = loss / self.grad_accum_steps

			self.grad_scaler.scale(loss).backward()

			if step_idx % self.grad_accum_steps == 0 or step_idx == len(self.train_loader):
				if self.grad_clip is not None:
					self.grad_scaler.unscale_(self.optimiser)
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.grad_clip))

				self.grad_scaler.step(self.optimiser)
				self.grad_scaler.update()
				self.optimiser.zero_grad(set_to_none=True)

			total_loss += float(raw_loss.item()) * batch_size
			total_samples += batch_size

			if pbar is not None:
				pbar.update(1)
				if step_idx % 10 == 0 or step_idx == num_steps:
					pbar.set_postfix_str(f"loss={raw_loss.item():.4f}", refresh=False)
			elif self.show_progress and (step_idx == 1 or step_idx % self.progress_log_interval == 0 or step_idx == num_steps):
				self.logger.info(
					"train step %s/%s | loss=%.4f",
					step_idx,
					num_steps,
					raw_loss.item(),
				)

		if pbar is not None:
			pbar.close()

		epoch_loss = total_loss / max(total_samples, 1)
		return {"loss": epoch_loss}

	@torch.no_grad()
	def val_epoch(self) -> dict[str, float]:
		self.model.eval()
		total_loss = 0.0
		total_samples = 0
		use_tqdm = self._use_tqdm
		num_steps = len(self.val_loader)
		all_logits: list[torch.Tensor] = []
		all_labels: list[torch.Tensor] = []

		pbar = None
		if use_tqdm:
			pbar = tqdm(
				total=num_steps,
				desc="val",
				leave=True,
				dynamic_ncols=True,
				mininterval=0.2,
				file=sys.stderr,
			)

		for step_idx, batch in enumerate(self.val_loader, start=1):
			inputs, labels = self._unpack_batch(batch)
			inputs = inputs.to(self.device, non_blocking=True)
			labels = labels.to(self.device, non_blocking=True)
			batch_size = labels.size(0)

			with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
				logits = self.model(inputs)
				loss = self.loss_fn(logits, labels)

			total_loss += float(loss.item()) * batch_size
			total_samples += batch_size
			all_logits.append(logits.detach())
			all_labels.append(labels.detach())

			if pbar is not None:
				pbar.update(1)
				if step_idx % 10 == 0 or step_idx == num_steps:
					pbar.set_postfix_str(f"loss={loss.item():.4f}", refresh=False)
			elif self.show_progress and (step_idx == 1 or step_idx % self.progress_log_interval == 0 or step_idx == num_steps):
				self.logger.info(
					"val step %s/%s | loss=%.4f",
					step_idx,
					num_steps,
					loss.item(),
				)

		if pbar is not None:
			pbar.close()

		epoch_loss = total_loss / max(total_samples, 1)
		epoch_logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, device=self.device)
		epoch_labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, device=self.device, dtype=torch.long)

		metrics = self._compute_eval_metrics(epoch_loss, epoch_logits, epoch_labels)
		return self._run_external_evaluator(epoch_logits, epoch_labels, metrics)

	def fit(self) -> dict[str, list[float]]:
		history: dict[str, list[float]] = {}
		max_epochs = int(self.config.train.epochs)

		for epoch in range(1, max_epochs + 1):
			train_metrics = self.train_epoch()
			val_metrics = self.val_epoch()

			self._step_scheduler(val_metrics["loss"])

			for key, value in train_metrics.items():
				history.setdefault(f"train_{key}", []).append(float(value))
			for key, value in val_metrics.items():
				history.setdefault(f"val_{key}", []).append(float(value))

			monitor_value = float(val_metrics.get(self._monitor_key.removeprefix("val_"), math.nan))
			if math.isnan(monitor_value):
				monitor_value = float(val_metrics["loss"])

			if self._is_improvement(monitor_value):
				self.best_score = monitor_value
				self._save_best_checkpoint(epoch=epoch, val_metrics=val_metrics)
				self.logger.info(
					"New best checkpoint saved at epoch=%s (%s=%.6f)",
					epoch,
					self._monitor_key,
					monitor_value,
				)

			self.logger.info(
				"Epoch %s/%s | train_loss=%.6f | val_loss=%.6f",
				epoch,
				max_epochs,
				train_metrics["loss"],
				val_metrics["loss"],
			)

			if self.early_stopping is not None:
				should_stop = self.early_stopping.step(metric=monitor_value, model=self.model)
				if should_stop:
					if getattr(self.early_stopping, "restore_best_weights", False):
						self.early_stopping.restore(self.model)
					break

		return history


__all__ = ["Trainer"]


if __name__ == "__main__":
	# Minimal smoke check with random data to validate interfaces.
	from torch.utils.data import DataLoader, TensorDataset

	torch.manual_seed(42)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	x_train = torch.randn(64, 3, 32, 32)
	y_train = torch.randint(0, 7, (64,))
	x_val = torch.randn(32, 3, 32, 32)
	y_val = torch.randint(0, 7, (32,))

	train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=8, shuffle=True)
	val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=8, shuffle=False)

	model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 7)).to(device)
	loss_fn = nn.CrossEntropyLoss()
	optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
	trainer = Trainer(
		model=model,
		loaders={"train": train_loader, "val": val_loader},
		loss_fn=loss_fn,
		optimiser=optimiser,
		scheduler=None,
		early_stopping=None,
		config=CFG,
	)

	history = trainer.fit()
	assert history["train_loss"], "Missing train history"
	assert history["val_loss"], "Missing val history"
	assert trainer.best_checkpoint_path.exists(), "Best checkpoint was not saved"
	print("Trainer smoke check passed.")
