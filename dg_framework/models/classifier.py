from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
	from dg_framework.models.layers import PassThrough
except ModuleNotFoundError:
	project_root = Path(__file__).resolve().parents[1]
	if str(project_root) not in sys.path:
		sys.path.append(str(project_root))
	from models.layers import PassThrough


def _cfg_get(config: Any, key: str, default: Any) -> Any:
	if isinstance(config, dict):
		return config.get(key, default)
	return getattr(config, key, default)


def _resolve_feature_tensor(features: Any) -> torch.Tensor:
	if isinstance(features, torch.Tensor):
		return features
	if isinstance(features, (list, tuple)) and features:
		last = features[-1]
		if isinstance(last, torch.Tensor):
			return last
	if isinstance(features, dict) and features:
		for value in features.values():
			if isinstance(value, torch.Tensor):
				return value
	raise TypeError("Backbone output must be a Tensor, tuple/list of Tensor, or dict with Tensor values")


def _flatten_features(features: torch.Tensor) -> torch.Tensor:
	if features.dim() == 1:
		return features.unsqueeze(0)
	if features.dim() == 2:
		return features
	if features.dim() == 4:
		return F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
	raise ValueError(f"Unsupported feature shape: {tuple(features.shape)}")


def _build_optional_layers(config: Any) -> nn.Module:
	layers_cfg = _cfg_get(config, "optional_layers", None)
	if layers_cfg is None:
		return PassThrough()
	if isinstance(layers_cfg, nn.Module):
		return layers_cfg
	if isinstance(layers_cfg, (list, tuple)):
		if not layers_cfg:
			return PassThrough()
		if not all(isinstance(layer, nn.Module) for layer in layers_cfg):
			raise TypeError("All elements of config.optional_layers must be nn.Module")
		return nn.Sequential(*layers_cfg)
	raise TypeError("config.optional_layers must be nn.Module or list/tuple of nn.Module")


def _build_head(feature_dim: int, num_classes: int, config: Any) -> nn.Module:
	head_type = _cfg_get(config, "head_type", "linear")
	head_depth = int(_cfg_get(config, "head_depth", 2))
	head_width = int(_cfg_get(config, "head_width", 512))

	if head_type == "linear":
		return nn.Linear(feature_dim, num_classes)

	if head_type != "mlp":
		raise ValueError(f"Unsupported head_type: {head_type}")
	if head_depth < 1:
		raise ValueError("head_depth must be >= 1")
	if head_width < 1:
		raise ValueError("head_width must be >= 1")
	if head_depth == 1:
		return nn.Linear(feature_dim, num_classes)

	layers: list[nn.Module] = []
	in_dim = feature_dim
	for _ in range(head_depth - 1):
		layers.append(nn.Linear(in_dim, head_width))
		layers.append(nn.ReLU(inplace=True))
		in_dim = head_width
	layers.append(nn.Linear(in_dim, num_classes))
	return nn.Sequential(*layers)


class DGClassifier(nn.Module):
	"""Compose backbone, optional intermediate layers, and classification head."""

	def __init__(
		self,
		backbone: nn.Module,
		feature_dim: int,
		num_classes: int,
		config: Any,
	) -> None:
		super().__init__()
		if feature_dim < 1:
			raise ValueError("feature_dim must be >= 1")
		if num_classes < 1:
			raise ValueError("num_classes must be >= 1")

		self.backbone = backbone
		self.feature_dim = int(feature_dim)
		self.num_classes = int(num_classes)
		self.config = config

		self.optional_layers = _build_optional_layers(config)
		self.head = _build_head(self.feature_dim, self.num_classes, config)

	def get_features(self, x: torch.Tensor) -> torch.Tensor:
		features = self.backbone(x)
		features = _resolve_feature_tensor(features)
		features = self.optional_layers(features)
		features = _resolve_feature_tensor(features)
		return _flatten_features(features)

	def forward(
		self,
		x: torch.Tensor,
		return_features: bool = False,
	) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
		features = self.get_features(x)
		logits = self.head(features)
		if return_features:
			return logits, features
		return logits


__all__ = ["DGClassifier"]


# Example usage and validation
if __name__ == "__main__":
	from types import SimpleNamespace

	try:
		from dg_framework.config import CFG
		from dg_framework.models.backbone import load_backbone
	except ModuleNotFoundError:
		project_root = Path(__file__).resolve().parents[1]
		if str(project_root) not in sys.path:
			sys.path.append(str(project_root))
		from config import CFG
		from models.backbone import load_backbone

	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
	)

	torch.manual_seed(42)
	batch_size = 2
	num_classes = 7
	x = torch.randn(batch_size, 3, 224, 224)

	backbone, feature_dim = load_backbone(
		name="resnet18",
		pretrained=False,
		freeze_up_to=None,
	)
	model = DGClassifier(
		backbone=backbone,
		feature_dim=feature_dim,
		num_classes=num_classes,
		config=CFG.model,
	)

	model.eval()
	logits = model(x)
	assert logits.shape == (batch_size, num_classes), "Invalid logits shape for linear head"

	logits_rf, features = model(x, return_features=True)
	assert logits_rf.shape == (batch_size, num_classes), "Invalid logits shape with return_features"
	assert features.shape == (batch_size, feature_dim), "Invalid feature shape"
	assert torch.allclose(logits, logits_rf), "Forward outputs mismatch between modes"

	mlp_config = SimpleNamespace(
		head_type="mlp",
		head_depth=3,
		head_width=256,
		optional_layers=nn.Sequential(PassThrough()),
	)
	mlp_model = DGClassifier(
		backbone=backbone,
		feature_dim=feature_dim,
		num_classes=num_classes,
		config=mlp_config,
	)
	mlp_model.eval()
	mlp_logits = mlp_model(x)
	assert mlp_logits.shape == (batch_size, num_classes), "Invalid logits shape for MLP head"

	print(
		"DGClassifier check passed. "
		f"feature_dim={feature_dim}, num_classes={num_classes}, "
		f"linear_head={model.head.__class__.__name__}, mlp_head={mlp_model.head.__class__.__name__}"
	)
