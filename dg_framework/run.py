from __future__ import annotations

import copy
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
	CosineAnnealingWarmRestarts,
	LinearLR,
	ReduceLROnPlateau,
	SequentialLR,
	StepLR,
)

try:
	from dg_framework.config import CFG, Config, validate_cfg
	from dg_framework.data.pacs import PACS
	from dg_framework.evaluation.evaluator import Evaluator
	from dg_framework.models.backbone import load_backbone
	from dg_framework.models.classifier import DGClassifier
	from dg_framework.training.early_stopping import EarlyStopping
	from dg_framework.training.losses import get_loss
	from dg_framework.training.trainer import Trainer
	from dg_framework.utils.experiment import setup_experiment
	from dg_framework.utils.logging_setup import get_logger
	from dg_framework.utils.reproducibility import set_seed
except ModuleNotFoundError:
	project_root = Path(__file__).resolve().parent
	if str(project_root) not in sys.path:
		sys.path.append(str(project_root))
	from config import CFG, Config, validate_cfg
	from data.pacs import PACS
	from evaluation.evaluator import Evaluator
	from models.backbone import load_backbone
	from models.classifier import DGClassifier
	from training.early_stopping import EarlyStopping
	from training.losses import get_loss
	from training.trainer import Trainer
	from utils.experiment import setup_experiment
	from utils.logging_setup import get_logger
	from utils.reproducibility import set_seed


class _WandbHook:
	"""Thin wandb wrapper with zero-overhead no-op path when disabled."""

	def __init__(self, cfg: Config, logger: logging.Logger) -> None:
		self.enabled = bool(cfg.experiment.use_wandb)
		self._logger = logger
		self._run: Any | None = None

		if not self.enabled:
			return

		try:
			import wandb  # type: ignore
		except ModuleNotFoundError:
			self.enabled = False
			self._logger.warning("wandb is enabled in config but package is not installed. Continuing without wandb.")
			return

		self._run = wandb.init(
			project=cfg.data.dataset_name.lower(),
			name=cfg.experiment.name,
			config=asdict(cfg),
			tags=list(cfg.experiment.tags),
		)

	def log(self, data: dict[str, float | int | str], step: int | None = None) -> None:
		if not self.enabled or self._run is None:
			return
		self._run.log(data, step=step)

	def finish(self) -> None:
		if not self.enabled or self._run is None:
			return
		self._run.finish()


def _config_summary_table(cfg: Config) -> str:
	rows = [
		("dataset", cfg.data.dataset_name),
		("target_domain", cfg.data.target_domain),
		("image_size", f"{cfg.data.image_size[0]}x{cfg.data.image_size[1]}"),
		("device_pref", cfg.train.device),
		("backbone", cfg.model.backbone_name),
		("head", cfg.model.head_type),
		("epochs", str(cfg.train.epochs)),
		("batch_size", str(cfg.train.batch_size)),
		("accum_steps", str(cfg.train.grad_accum_steps)),
		("effective_batch", str(cfg.train.batch_size * cfg.train.grad_accum_steps)),
		("optimiser", cfg.train.optimiser),
		("lr", f"{cfg.train.lr:.2e}"),
		("scheduler", cfg.train.scheduler.scheduler_type),
		("warmup_epochs", str(cfg.train.lr_warmup_epochs)),
		("amp", str(cfg.train.amp)),
		("progress", str(getattr(cfg.train, "show_progress", True))),
		("ema", str(cfg.train.ema.enabled)),
		("tta", str(cfg.evaluation.tta.enabled)),
		("wandb", str(cfg.experiment.use_wandb)),
		("experiment", cfg.experiment.name),
	]

	key_width = max(len(k) for k, _ in rows)
	val_width = max(len(v) for _, v in rows)
	border = "+-" + "-" * key_width + "-+-" + "-" * val_width + "-+"

	lines = ["RUN CONFIG SUMMARY", border]
	for key, value in rows:
		lines.append(f"| {key:<{key_width}} | {value:<{val_width}} |")
	lines.append(border)
	return "\n".join(lines)


def _count_model_parameters(model: torch.nn.Module) -> tuple[int, int, int]:
	total = sum(param.numel() for param in model.parameters())
	trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
	frozen = total - trainable
	return total, trainable, frozen


def _format_dataset_summary(
	dataset: Any,
	train_loader: Any,
	val_loader: Any,
	test_loader: Any,
	cfg: Config,
) -> str:
	train_size = len(train_loader.dataset)
	val_size = len(val_loader.dataset)
	test_size = len(test_loader.dataset)

	domain_names = dataset.get_domain_names() if hasattr(dataset, "get_domain_names") else []
	class_names = dataset.get_class_names() if hasattr(dataset, "get_class_names") else []
	train_domains = [name for name in domain_names if name != cfg.data.target_domain]

	lines = [
		"DATASET SUMMARY",
		f"- dataset: {cfg.data.dataset_name}",
		f"- root: {cfg.data.root_path}",
		f"- target_domain: {cfg.data.target_domain}",
		f"- train_domains: {train_domains}",
		f"- all_domains: {domain_names}",
		f"- num_classes: {len(class_names)}",
		f"- class_names: {class_names}",
		f"- samples: train={train_size}, val={val_size}, test={test_size}, total={train_size + val_size + test_size}",
		f"- batches/epoch: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}",
	]

	if hasattr(train_loader.dataset, "samples"):
		try:
			train_label_counts = Counter(sample.label for sample in train_loader.dataset.samples)
			ordered_counts = []
			for idx, count in sorted(train_label_counts.items(), key=lambda item: item[0]):
				class_name = class_names[idx] if idx < len(class_names) else str(idx)
				ordered_counts.append(f"{class_name}:{count}")
			lines.append(f"- train_class_distribution: {' | '.join(ordered_counts)}")
		except Exception:
			pass

	return "\n".join(lines)


def _describe_head(head_module: torch.nn.Module) -> str:
	if isinstance(head_module, torch.nn.Linear):
		return f"Linear({head_module.in_features}->{head_module.out_features})"
	if isinstance(head_module, torch.nn.Sequential):
		parts: list[str] = []
		for layer in head_module:
			if isinstance(layer, torch.nn.Linear):
				parts.append(f"Linear({layer.in_features}->{layer.out_features})")
			else:
				parts.append(layer.__class__.__name__)
		return "Sequential[" + " -> ".join(parts) + "]"
	return head_module.__class__.__name__


def _format_model_diagram(model: DGClassifier, cfg: Config) -> str:
	head_desc = _describe_head(model.head)

	lines = [
		"MODEL BLOCK DIAGRAM",
		"[Input: Bx3xHxW]",
		"      |",
		f"[Backbone: {cfg.model.backbone_name}]",
		"      |",
		f"[Optional Layers: {model.optional_layers.__class__.__name__}]",
		"      |",
		f"[Feature Vector: Bx{model.feature_dim}]",
		"      |",
		f"[Head: {head_desc}]",
		"      |",
		f"[Logits: Bx{model.num_classes}]",
	]
	return "\n".join(lines)


def _format_runtime_summary(
	model: DGClassifier,
	cfg: Config,
	device: torch.device,
	train_loader: Any,
) -> str:
	total_params, trainable_params, frozen_params = _count_model_parameters(model)
	effective_batch = cfg.train.batch_size * cfg.train.grad_accum_steps
	steps_per_epoch = len(train_loader)
	total_updates = steps_per_epoch * cfg.train.epochs

	lines = [
		"RUNTIME SUMMARY",
		f"- device_selected: {device}",
		f"- requested_device: {cfg.train.device}",
		f"- cuda_available: {torch.cuda.is_available()}",
		f"- mps_available: {_is_mps_available()}",
		f"- amp_enabled: {cfg.train.amp and device.type == 'cuda'}",
		f"- batch_size: {cfg.train.batch_size}",
		f"- grad_accum_steps: {cfg.train.grad_accum_steps}",
		f"- effective_batch_size: {effective_batch}",
		f"- steps_per_epoch: {steps_per_epoch}",
		f"- max_epochs: {cfg.train.epochs}",
		f"- optimiser: {cfg.train.optimiser}",
		f"- scheduler: {cfg.train.scheduler.scheduler_type}",
		f"- total_parameter_updates_estimate: {total_updates}",
		f"- model_params_total: {total_params:,}",
		f"- model_params_trainable: {trainable_params:,}",
		f"- model_params_frozen: {frozen_params:,}",
	]
	return "\n".join(lines)


def _is_mps_available() -> bool:
	if not hasattr(torch.backends, "mps"):
		return False
	return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())


def _select_device(cfg: Config, logger: logging.Logger) -> torch.device:
	requested = str(cfg.train.device).lower()

	if requested == "auto":
		if torch.cuda.is_available():
			device = torch.device("cuda")
		elif _is_mps_available():
			device = torch.device("mps")
		else:
			device = torch.device("cpu")
		logger.info("Device auto-selection resolved to: %s", device)
		return device

	if requested == "cpu":
		return torch.device("cpu")

	if requested == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("train.device='cuda' requested, but CUDA is not available")
		return torch.device("cuda")

	if requested == "mps":
		if not _is_mps_available():
			raise RuntimeError("train.device='mps' requested, but Apple MPS is not available")
		return torch.device("mps")

	raise ValueError(f"Unsupported train.device value: {cfg.train.device}")


def _build_data_loaders(cfg: Config, device: torch.device):
	dataset = PACS(cfg)
	target_domain = cfg.data.target_domain
	train_domains = [name for name in dataset.get_domain_names() if name != target_domain]

	train_loader, val_loader, test_loader = dataset.get_loaders(
		train_domains=train_domains,
		test_domain=target_domain,
		batch_size=cfg.train.batch_size,
		num_workers=cfg.data.num_workers,
		pin_memory=(device.type == "cuda"),
		drop_last=False,
	)
	return dataset, train_loader, val_loader, test_loader


def _build_model(cfg: Config, num_classes: int, device: torch.device) -> DGClassifier:
	backbone, feature_dim = load_backbone(
		name=cfg.model.backbone_name,
		pretrained=cfg.model.pretrained,
		freeze_up_to=cfg.model.freeze_up_to,
	)
	model = DGClassifier(
		backbone=backbone,
		feature_dim=feature_dim,
		num_classes=num_classes,
		config=cfg.model,
	)
	return model.to(device)


def _build_optimiser(cfg: Config, model: torch.nn.Module) -> Optimizer:
	trainable_params = [p for p in model.parameters() if p.requires_grad]
	if not trainable_params:
		raise ValueError("No trainable parameters found in model")

	if cfg.train.optimiser == "adamw":
		return torch.optim.AdamW(
			trainable_params,
			lr=cfg.train.lr,
			weight_decay=cfg.train.weight_decay,
		)

	if cfg.train.optimiser == "sgd":
		return torch.optim.SGD(
			trainable_params,
			lr=cfg.train.lr,
			momentum=0.9,
			weight_decay=cfg.train.weight_decay,
			nesterov=True,
		)

	raise ValueError(f"Unsupported optimiser: {cfg.train.optimiser}")


def _build_scheduler(cfg: Config, optimiser: Optimizer):
	scheduler_cfg = cfg.train.scheduler
	base_scheduler: Any | None

	if scheduler_cfg.scheduler_type == "none":
		return None
	if scheduler_cfg.scheduler_type == "step":
		base_scheduler = StepLR(
			optimiser,
			step_size=max(1, scheduler_cfg.step_size),
			gamma=scheduler_cfg.gamma,
		)
	elif scheduler_cfg.scheduler_type == "plateau":
		base_scheduler = ReduceLROnPlateau(
			optimiser,
			mode="min",
			factor=scheduler_cfg.plateau_factor,
			patience=max(1, scheduler_cfg.plateau_patience),
		)
	elif scheduler_cfg.scheduler_type == "cosine":
		t0 = max(1, cfg.train.epochs - max(0, cfg.train.lr_warmup_epochs))
		base_scheduler = CosineAnnealingWarmRestarts(
			optimiser,
			T_0=t0,
			T_mult=1,
			eta_min=scheduler_cfg.min_lr,
		)
	else:
		raise ValueError(f"Unsupported scheduler_type: {scheduler_cfg.scheduler_type}")

	warmup_epochs = max(0, int(cfg.train.lr_warmup_epochs))
	if warmup_epochs == 0 or scheduler_cfg.scheduler_type == "plateau":
		return base_scheduler

	warmup = LinearLR(
		optimiser,
		start_factor=1e-3,
		end_factor=1.0,
		total_iters=warmup_epochs,
	)
	return SequentialLR(
		optimiser,
		schedulers=[warmup, base_scheduler],
		milestones=[warmup_epochs],
	)


def _build_early_stopping(cfg: Config) -> EarlyStopping:
	monitor_mode = "max" if "accuracy" in set(cfg.evaluation.metrics) else "min"
	patience = int(getattr(cfg.train, "early_stopping_patience", 5))
	min_delta = float(getattr(cfg.train, "early_stopping_min_delta", 0.0))
	restore = bool(getattr(cfg.train, "restore_best_weights", True))

	return EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		mode=monitor_mode,
		restore_best_weights=restore,
	)


def _normalise_metric_names(cfg: Config) -> list[str]:
	resolved: list[str] = []
	for metric_name in cfg.evaluation.metrics:
		if metric_name == "f1":
			resolved.append("f1_macro")
		else:
			resolved.append(metric_name)

	if any(k > 1 for k in cfg.evaluation.top_k) and "top_k_accuracy" not in resolved:
		resolved.append("top_k_accuracy")

	# Keep calibration metrics explicit in the final report.
	if "expected_calibration_error" not in resolved and "ece" not in resolved:
		resolved.append("expected_calibration_error")
	if "brier_score" not in resolved:
		resolved.append("brier_score")

	ordered = []
	seen = set()
	for name in resolved:
		if name in seen:
			continue
		seen.add(name)
		ordered.append(name)
	return ordered


def run(cfg: Config) -> tuple[Path, dict[str, Any]]:
	validate_cfg(cfg)
	set_seed(cfg.train.seed, deterministic=cfg.train.deterministic)

	experiment_path = setup_experiment(cfg)
	logger = get_logger("dg_framework", experiment_path, cfg.experiment.log_level)
	logger.info("%s", _config_summary_table(cfg))

	wandb_hook = _WandbHook(cfg=cfg, logger=logger)
	device = _select_device(cfg=cfg, logger=logger)
	logger.info("Using device: %s (requested=%s)", device, cfg.train.device)

	logger.info("Building data loaders...")
	dataset, train_loader, val_loader, test_loader = _build_data_loaders(cfg=cfg, device=device)
	logger.info("%s", _format_dataset_summary(dataset, train_loader, val_loader, test_loader, cfg))
	logger.info("Building model...")
	model = _build_model(cfg=cfg, num_classes=dataset.num_classes, device=device)
	logger.info("%s", _format_model_diagram(model, cfg))
	logger.info("%s", _format_runtime_summary(model, cfg, device, train_loader))

	logger.info("Building loss, optimiser, scheduler, and early stopping...")
	loss_fn = get_loss(
		cfg.train.loss_name,
		label_smoothing=cfg.train.label_smoothing,
	).to(device)
	optimiser = _build_optimiser(cfg=cfg, model=model)
	scheduler = _build_scheduler(cfg=cfg, optimiser=optimiser)
	early_stopping = _build_early_stopping(cfg=cfg)

	trainer = Trainer(
		model=model,
		loaders={"train": train_loader, "val": val_loader},
		loss_fn=loss_fn,
		optimiser=optimiser,
		scheduler=scheduler,
		early_stopping=early_stopping,
		config=cfg,
	)
	logger.info("Starting training loop...")
	history = trainer.fit()

	for epoch_idx in range(len(history.get("train_loss", []))):
		epoch_payload: dict[str, float] = {}
		for key, values in history.items():
			epoch_payload[key] = float(values[epoch_idx])
		wandb_hook.log(epoch_payload, step=epoch_idx + 1)

	eval_metrics = _normalise_metric_names(cfg)
	evaluator = Evaluator(
		model=model,
		loader=test_loader,
		metrics=eval_metrics,
		device=device,
		config=cfg,
		experiment_path=experiment_path,
	)
	logger.info("Running evaluation...")
	overall_test_metrics = evaluator.evaluate()
	per_domain_metrics: dict[str, dict[str, float]] = {}
	if cfg.evaluation.per_domain_eval:
		logger.info("Running per-domain evaluation...")
		per_domain_metrics = evaluator.evaluate_per_domain(test_loader)

	results: dict[str, Any] = {
		"experiment": cfg.experiment.name,
		"target_domain": cfg.data.target_domain,
		"device_requested": cfg.train.device,
		"device": str(device),
		"history": history,
		"test_metrics": overall_test_metrics,
		"per_domain_metrics": per_domain_metrics,
		"best_checkpoint": str(trainer.best_checkpoint_path),
	}

	results_path = experiment_path / "results.json"
	results_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
	logger.info("Saved results to %s", results_path)

	wandb_hook.log({f"test/{k}": float(v) for k, v in overall_test_metrics.items()})
	wandb_hook.finish()

	return experiment_path, results


def _run_integration_smoke_test() -> None:
	cfg = copy.deepcopy(CFG)
	cfg.experiment.name = f"{CFG.experiment.name}_integration"
	cfg.train.epochs = 1
	cfg.train.batch_size = max(4, min(16, CFG.train.batch_size))
	cfg.train.lr_warmup_epochs = 0
	cfg.model.backbone_name = "resnet18"
	cfg.model.pretrained = False
	cfg.data.num_workers = 0
	cfg.experiment.use_wandb = False
	cfg.experiment.save_confusion_matrix = True
	cfg.experiment.export_embeddings = True

	experiment_path, _ = run(cfg)
	results_path = experiment_path / "results.json"
	config_copy_path = experiment_path / "config.py"
	figure_paths = sorted((experiment_path / "figures").glob("*.png"))

	assert results_path.is_file(), "Missing results.json after integration run"
	assert config_copy_path.is_file(), "Missing copied config.py in experiment folder"
	assert figure_paths, "No confusion matrix PNG generated in figures folder"

	results = json.loads(results_path.read_text(encoding="utf-8"))
	assert "test_metrics" in results and results["test_metrics"], "Missing test_metrics in results.json"

	print(
		"Integration check passed. "
		f"experiment_path={experiment_path}, "
		f"figures={len(figure_paths)}, "
		f"metrics={sorted(results['test_metrics'].keys())}"
	)


if __name__ == "__main__":
	run_mode = os.environ.get("DG_RUN_MODE", "train").strip().lower()
	if run_mode == "integration":
		_run_integration_smoke_test()
	else:
		run(CFG)
