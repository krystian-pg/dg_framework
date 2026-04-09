from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pprint import pprint
from typing import Literal


@dataclass
class DataConfig:
	dataset_name: str = "PACS"
	root_path: str = "pacs_data/pacs_data"
	target_domain: str = "sketch"
	train_val_split: float = 0.8
	image_size: tuple[int, int] = (224, 224)
	num_workers: int = 8


@dataclass
class ModelConfig:
	backbone_name: str = "resnet50"
	pretrained: bool = True
	freeze_up_to: int | None = None
	use_custom_layers: bool = False
	head_type: Literal["linear", "mlp"] = "linear"
	head_depth: int = 2
	head_width: int = 512


@dataclass
class SchedulerConfig:
	scheduler_type: Literal["none", "cosine", "step", "plateau"] = "cosine"
	min_lr: float = 1e-6
	step_size: int = 10
	gamma: float = 0.1
	plateau_factor: float = 0.5
	plateau_patience: int = 3


@dataclass
class EMAConfig:
	enabled: bool = False
	decay: float = 0.999


@dataclass
class TrainConfig:
	epochs: int = 30
	batch_size: int = 32
	lr: float = 3e-4
	weight_decay: float = 1e-4
	optimiser: Literal["adamw", "sgd"] = "adamw"
	loss_name: str = "cross_entropy"
	label_smoothing: float = 0.0
	lr_warmup_epochs: int = 3
	scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
	grad_clip: float | None = 1.0
	grad_accum_steps: int = 1
	amp: bool = True
	device: Literal["auto", "cpu", "cuda", "mps"] = "mps"
	show_progress: bool = True
	progress_log_interval: int = 20
	deterministic: bool = True
	ema: EMAConfig = field(default_factory=EMAConfig)
	seed: int = 42


@dataclass
class TTAConfig:
	enabled: bool = False
	n_augments: int = 4


@dataclass
class EvalConfig:
	metrics: list[str] = field(default_factory=lambda: ["accuracy", "f1", "ece"])
	top_k: tuple[int, ...] = (1, 5)
	ece_bins: int = 15
	tta: TTAConfig = field(default_factory=TTAConfig)
	per_domain_eval: bool = True


@dataclass
class ExperimentConfig:
	name: str = "pacs_baseline_2"
	output_root: str = "../outputs"
	use_wandb: bool = False
	log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
	save_confusion_matrix: bool = True
	export_embeddings: bool = False
	tags: list[str] = field(default_factory=lambda: ["pacs", "dg"])


@dataclass
class Config:
	data: DataConfig = field(default_factory=DataConfig)
	model: ModelConfig = field(default_factory=ModelConfig)
	train: TrainConfig = field(default_factory=TrainConfig)
	evaluation: EvalConfig = field(default_factory=EvalConfig)
	experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def validate_cfg(cfg: Config) -> None:
	if not isinstance(cfg.data.dataset_name, str) or not cfg.data.dataset_name:
		raise ValueError("data.dataset_name must be a non-empty string")
	if not 0.0 < cfg.data.train_val_split < 1.0:
		raise ValueError("data.train_val_split must be in (0, 1)")
	if len(cfg.data.image_size) != 2 or any(x <= 0 for x in cfg.data.image_size):
		raise ValueError("data.image_size must be a positive (H, W) tuple")
	if cfg.model.freeze_up_to is not None and cfg.model.freeze_up_to < 0:
		raise ValueError("model.freeze_up_to must be >= 0 or None")
	if cfg.model.head_depth < 1 or cfg.model.head_width < 1:
		raise ValueError("model.head_depth and model.head_width must be >= 1")
	if cfg.train.epochs < 1 or cfg.train.batch_size < 1:
		raise ValueError("train.epochs and train.batch_size must be >= 1")
	if cfg.train.lr <= 0 or cfg.train.weight_decay < 0:
		raise ValueError("train.lr must be > 0 and train.weight_decay must be >= 0")
	if not 0.0 <= cfg.train.label_smoothing < 1.0:
		raise ValueError("train.label_smoothing must be in [0, 1)")
	if cfg.train.grad_accum_steps < 1:
		raise ValueError("train.grad_accum_steps must be >= 1")
	if cfg.train.progress_log_interval < 1:
		raise ValueError("train.progress_log_interval must be >= 1")
	if cfg.train.device not in {"auto", "cpu", "cuda", "mps"}:
		raise ValueError("train.device must be one of: 'auto', 'cpu', 'cuda', 'mps'")
	if cfg.train.grad_clip is not None and cfg.train.grad_clip <= 0:
		raise ValueError("train.grad_clip must be > 0 or None")
	if cfg.train.ema.decay <= 0.0 or cfg.train.ema.decay >= 1.0:
		raise ValueError("train.ema.decay must be in (0, 1)")
	if cfg.evaluation.ece_bins < 1:
		raise ValueError("evaluation.ece_bins must be >= 1")
	if not cfg.evaluation.top_k or any(k < 1 for k in cfg.evaluation.top_k):
		raise ValueError("evaluation.top_k must contain positive integer values")
	if cfg.evaluation.tta.enabled and cfg.evaluation.tta.n_augments < 1:
		raise ValueError("evaluation.tta.n_augments must be >= 1 when TTA is enabled")


CFG = Config(
	data=DataConfig(),
	model=ModelConfig(),
	train=TrainConfig(),
	evaluation=EvalConfig(),
	experiment=ExperimentConfig(),
)

# Example usage and validation
if __name__ == "__main__":
	validate_cfg(CFG)
	print("Config object:\n")
	pprint(CFG)
	print("\nConfig as dict:\n")
	pprint(asdict(CFG))
