from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

try:
	from dg_framework.config import CFG, Config
	from dg_framework.evaluation.metrics import compute_all
	from dg_framework.utils.experiment import setup_experiment
except ModuleNotFoundError:
	project_root = Path(__file__).resolve().parents[1]
	if str(project_root) not in sys.path:
		sys.path.append(str(project_root))
	from config import CFG, Config
	from evaluation.metrics import compute_all
	from utils.experiment import setup_experiment


class Evaluator:
	"""Runs model evaluation and returns a full metric report."""

	def __init__(
		self,
		model: torch.nn.Module,
		loader: DataLoader,
		metrics: list[str],
		device: torch.device | str,
		config: Config | None = None,
		experiment_path: Path | None = None,
	) -> None:
		self.model = model
		self.loader = loader
		self.metrics = metrics
		self.device = torch.device(device)
		self.config = config or CFG

		self.model.to(self.device)
		self.experiment_path = experiment_path or setup_experiment(self.config)
		self.figures_path = self.experiment_path / "figures"
		self.embeddings_path = self.experiment_path / "embeddings"
		self.figures_path.mkdir(parents=True, exist_ok=True)
		self.embeddings_path.mkdir(parents=True, exist_ok=True)

		self.save_confusion_matrix = bool(getattr(self.config.experiment, "save_confusion_matrix", True))
		self.export_embeddings = bool(getattr(self.config.experiment, "export_embeddings", False))
		self.tta_enabled = bool(getattr(self.config.evaluation.tta, "enabled", False))
		self.tta_n_augments = max(1, int(getattr(self.config.evaluation.tta, "n_augments", 1)))

	@staticmethod
	def _extract_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, Any | None, Any | None]:
		if not isinstance(batch, (tuple, list)) or len(batch) < 2:
			raise ValueError("Expected batch structure: (inputs, labels, [domain], [paths], ...)")
		inputs = batch[0]
		labels = batch[1]
		domain_labels = batch[2] if len(batch) > 2 else None
		paths = batch[3] if len(batch) > 3 else None
		return inputs, labels, domain_labels, paths

	@staticmethod
	def _to_numpy_labels(labels: torch.Tensor) -> np.ndarray:
		return labels.detach().cpu().numpy().astype(np.int64, copy=False)

	@staticmethod
	def _to_numpy_probabilities(logits: torch.Tensor) -> np.ndarray:
		probs = torch.softmax(logits, dim=1)
		return probs.detach().cpu().numpy().astype(np.float64, copy=False)

	@staticmethod
	def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
		cm = np.zeros((n_classes, n_classes), dtype=np.int64)
		np.add.at(cm, (y_true, y_pred), 1)
		return cm

	@staticmethod
	def _sanitize_filename(value: str) -> str:
		value = value.strip().lower()
		value = re.sub(r"[^a-z0-9_-]+", "_", value)
		return value.strip("_") or "domain"

	@staticmethod
	def _domain_keys(domain_labels: Any, paths: Any, batch_size: int) -> list[str]:
		if domain_labels is None:
			raise ValueError("Domain labels are required for evaluate_per_domain")

		if torch.is_tensor(domain_labels):
			domain_values = domain_labels.detach().cpu().tolist()
		else:
			domain_values = list(domain_labels)

		if len(domain_values) != batch_size:
			raise ValueError("Domain labels length must match batch size")

		if paths is not None:
			path_values = list(paths)
			if len(path_values) == batch_size:
				resolved: list[str] = []
				for idx, path_value in enumerate(path_values):
					try:
						domain_name = Path(str(path_value)).parents[1].name
					except IndexError:
						domain_name = f"domain_{domain_values[idx]}"
					resolved.append(domain_name)
				return resolved

		return [f"domain_{value}" for value in domain_values]

	def _save_confusion_matrix_png(self, y_true: np.ndarray, y_prob: np.ndarray, file_name: str) -> Path:
		n_classes = y_prob.shape[1]
		y_pred = np.argmax(y_prob, axis=1)
		cm = self._confusion_matrix(y_true=y_true, y_pred=y_pred, n_classes=n_classes)

		row_sum = cm.sum(axis=1, keepdims=True).astype(np.float64)
		normalized = cm.astype(np.float64) / np.maximum(row_sum, 1.0)
		img_array = (normalized * 255.0).clip(0, 255).astype(np.uint8)

		image = Image.fromarray(img_array, mode="L").resize(
			(max(1, n_classes) * 40, max(1, n_classes) * 40),
			resample=Image.NEAREST,
		)
		output_path = self.figures_path / file_name
		image.save(output_path, format="PNG")
		return output_path

	def _forward_probs(self, inputs: torch.Tensor) -> torch.Tensor:
		if not self.tta_enabled:
			logits = self.model(inputs)
			return torch.softmax(logits, dim=1)

		augmented_probs: list[torch.Tensor] = []
		for aug_idx in range(self.tta_n_augments):
			if aug_idx % 2 == 1:
				aug_inputs = torch.flip(inputs, dims=[3])
			else:
				aug_inputs = inputs

			logits = self.model(aug_inputs)
			augmented_probs.append(torch.softmax(logits, dim=1))

		return torch.stack(augmented_probs, dim=0).mean(dim=0)

	def _forward_features(self, inputs: torch.Tensor) -> torch.Tensor | None:
		try:
			_, features = self.model(inputs, return_features=True)
		except TypeError:
			return None
		if not torch.is_tensor(features):
			return None
		return features

	def _maybe_export_embeddings(
		self,
		embeddings: list[torch.Tensor],
		y_true: np.ndarray,
		y_prob: np.ndarray,
		file_name: str,
	) -> Path | None:
		if not self.export_embeddings or not embeddings:
			return None

		emb = torch.cat(embeddings, dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
		output_path = self.embeddings_path / file_name
		np.savez_compressed(output_path, embeddings=emb, y_true=y_true, y_prob=y_prob)
		return output_path

	@torch.no_grad()
	def evaluate(self) -> dict[str, float]:
		self.model.eval()
		all_probs: list[torch.Tensor] = []
		all_labels: list[torch.Tensor] = []
		all_embeddings: list[torch.Tensor] = []

		for batch in self.loader:
			inputs, labels, _, _ = self._extract_batch(batch)
			inputs = inputs.to(self.device, non_blocking=True)
			labels = labels.to(self.device, non_blocking=True)

			probs = self._forward_probs(inputs)
			all_probs.append(probs.detach().cpu())
			features = self._forward_features(inputs)
			if features is not None:
				all_embeddings.append(features.detach().cpu())
			all_labels.append(labels.detach().cpu())

		if not all_probs:
			raise ValueError("Loader is empty; cannot evaluate")

		epoch_probs = torch.cat(all_probs, dim=0)
		epoch_labels = torch.cat(all_labels, dim=0)

		y_true = self._to_numpy_labels(epoch_labels)
		y_prob = epoch_probs.numpy().astype(np.float64, copy=False)
		results = compute_all(self.metrics, y_true=y_true, y_prob=y_prob)
		self._maybe_export_embeddings(
			embeddings=all_embeddings,
			y_true=y_true,
			y_prob=y_prob,
			file_name="eval_embeddings.npz",
		)

		if self.save_confusion_matrix:
			self._save_confusion_matrix_png(y_true=y_true, y_prob=y_prob, file_name="confusion_matrix.png")
		return results

	@torch.no_grad()
	def evaluate_per_domain(self, loader_with_domain_labels: DataLoader) -> dict[str, dict[str, float]]:
		self.model.eval()
		per_domain_probs: dict[str, list[torch.Tensor]] = defaultdict(list)
		per_domain_labels: dict[str, list[torch.Tensor]] = defaultdict(list)
		per_domain_embeddings: dict[str, list[torch.Tensor]] = defaultdict(list)

		for batch in loader_with_domain_labels:
			inputs, labels, domain_labels, paths = self._extract_batch(batch)
			inputs = inputs.to(self.device, non_blocking=True)
			labels = labels.to(self.device, non_blocking=True)

			probs = self._forward_probs(inputs)
			features = self._forward_features(inputs)
			domain_keys = self._domain_keys(domain_labels=domain_labels, paths=paths, batch_size=labels.size(0))

			for idx, domain_key in enumerate(domain_keys):
				per_domain_probs[domain_key].append(probs[idx].detach().cpu().unsqueeze(0))
				per_domain_labels[domain_key].append(labels[idx].detach().cpu().unsqueeze(0))
				if features is not None:
					per_domain_embeddings[domain_key].append(features[idx].detach().cpu().unsqueeze(0))

		if not per_domain_probs:
			raise ValueError("No domain samples found in loader; cannot compute per-domain metrics")

		report: dict[str, dict[str, float]] = {}
		for domain_key in sorted(per_domain_probs.keys()):
			domain_probs = torch.cat(per_domain_probs[domain_key], dim=0)
			domain_labels = torch.cat(per_domain_labels[domain_key], dim=0)
			y_true = self._to_numpy_labels(domain_labels)
			y_prob = domain_probs.numpy().astype(np.float64, copy=False)
			report[domain_key] = compute_all(self.metrics, y_true=y_true, y_prob=y_prob)
			self._maybe_export_embeddings(
				embeddings=per_domain_embeddings.get(domain_key, []),
				y_true=y_true,
				y_prob=y_prob,
				file_name=f"eval_embeddings_{self._sanitize_filename(domain_key)}.npz",
			)

			if self.save_confusion_matrix:
				safe_name = self._sanitize_filename(domain_key)
				self._save_confusion_matrix_png(
					y_true=y_true,
					y_prob=y_prob,
					file_name=f"confusion_matrix_{safe_name}.png",
				)

		return report


__all__ = ["Evaluator"]


# Example usage and validation
if __name__ == "__main__":
	from torch.utils.data import TensorDataset

	torch.manual_seed(42)
	rng = np.random.default_rng(42)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	n_classes = 7

	inputs = torch.randn(40, 3, 32, 32)
	labels = torch.from_numpy(rng.integers(0, n_classes, size=40, dtype=np.int64))
	domains = torch.from_numpy(rng.integers(0, 4, size=40, dtype=np.int64))
	paths = [f"/tmp/domain_{int(d.item())}/class_x/img_{idx}.jpg" for idx, d in enumerate(domains)]

	loader = DataLoader(TensorDataset(inputs, labels), batch_size=8, shuffle=False)
	loader_with_domains = DataLoader(list(zip(inputs, labels, domains, paths, strict=False)), batch_size=8, shuffle=False)

	model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, n_classes)).to(device)
	metrics = ["accuracy", "f1_macro", "brier_score", "expected_calibration_error", "top_k_accuracy"]
	evaluator = Evaluator(model=model, loader=loader, metrics=metrics, device=device)

	overall = evaluator.evaluate()
	assert set(overall.keys()) == set(metrics), "Overall metric keys mismatch"

	per_domain = evaluator.evaluate_per_domain(loader_with_domain_labels=loader_with_domains)
	assert per_domain, "Per-domain report should not be empty"
	for domain_name, domain_metrics in per_domain.items():
		assert set(domain_metrics.keys()) == set(metrics), f"Per-domain keys mismatch for {domain_name}"

	assert (evaluator.figures_path / "confusion_matrix.png").is_file(), "Missing global confusion matrix PNG"
	print(
		"Evaluator check passed. "
		f"overall_metrics={sorted(overall.keys())}, "
		f"domains={sorted(per_domain.keys())[:4]}"
	)
