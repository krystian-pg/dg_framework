from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
	from dg_framework.models.layers import MixStyle
except ModuleNotFoundError:
	project_root = Path(__file__).resolve().parents[1]
	if str(project_root) not in sys.path:
		sys.path.append(str(project_root))
	from models.layers import MixStyle

try:
	import timm
except ModuleNotFoundError as exc:
	raise ModuleNotFoundError(
		"Missing dependency 'timm'. Install it with: pip install timm"
	) from exc


def _infer_feature_dim(backbone: nn.Module) -> int:
	feature_dim = getattr(backbone, "num_features", None)
	if isinstance(feature_dim, int) and feature_dim > 0:
		return feature_dim

	raise ValueError(
		"Could not infer backbone feature_dim. The selected model does not expose a valid "
		"'num_features' attribute."
	)


def _group_parameters(backbone: nn.Module) -> OrderedDict[str, list[nn.Parameter]]:
	groups: OrderedDict[str, list[nn.Parameter]] = OrderedDict()
	for param_name, param in backbone.named_parameters():
		group_name = param_name.split(".", 1)[0]
		groups.setdefault(group_name, []).append(param)
	return groups


def _cfg_get(config: Any, key: str, default: Any) -> Any:
	if config is None:
		return default
	if isinstance(config, dict):
		return config.get(key, default)
	return getattr(config, key, default)


def _attach_mixstyle_hooks(
	backbone: nn.Module,
	model_cfg: Any,
	logger: logging.Logger,
) -> list[str]:
	enabled = bool(_cfg_get(model_cfg, "mixstyle_enabled", False))
	if not enabled:
		return []

	p = float(_cfg_get(model_cfg, "mixstyle_p", 0.5))
	alpha = float(_cfg_get(model_cfg, "mixstyle_alpha", 0.1))
	eps = float(_cfg_get(model_cfg, "mixstyle_eps", 1e-6))
	layer_names = [str(name) for name in _cfg_get(model_cfg, "mixstyle_layers", ())]

	if not layer_names:
		logger.warning("MixStyle enabled but no layer names were provided; skipping MixStyle hooks")
		return []

	module_map = dict(backbone.named_modules())
	mixstyle_blocks = nn.ModuleDict()
	handles: list[Any] = []
	attached_layers: list[str] = []

	for idx, layer_name in enumerate(layer_names):
		if layer_name not in module_map:
			logger.warning("MixStyle target '%s' not found in backbone modules; skipping", layer_name)
			continue

		target_module = module_map[layer_name]
		module_key = f"layer_{idx}_{layer_name.replace('.', '_')}"
		mixstyle_blocks[module_key] = MixStyle(p=p, alpha=alpha, eps=eps)

		def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, key: str = module_key) -> Any:
			if isinstance(output, torch.Tensor) and output.dim() == 4:
				return mixstyle_blocks[key](output)
			return output

		handles.append(target_module.register_forward_hook(_hook))
		attached_layers.append(layer_name)

	if not attached_layers:
		logger.warning("MixStyle enabled but no hooks were attached")
		return []

	backbone.add_module("mixstyle_blocks", mixstyle_blocks)
	setattr(backbone, "_mixstyle_hook_handles", handles)
	logger.info(
		"MixStyle enabled: layers=%s, p=%.3f, alpha=%.3f, eps=%g",
		attached_layers,
		p,
		alpha,
		eps,
	)
	return attached_layers


def _freeze_parameter_groups(
	backbone: nn.Module,
	freeze_up_to: int | None,
	logger: logging.Logger,
) -> tuple[list[str], list[str]]:
	groups = _group_parameters(backbone)
	group_names = list(groups.keys())

	if freeze_up_to is None:
		logger.info("Backbone freezing: freeze_up_to=None, all layers trainable")
		return [], group_names

	if freeze_up_to < 0:
		raise ValueError("freeze_up_to must be >= 0 or None")

	freeze_count = min(freeze_up_to, len(group_names))
	frozen_groups = group_names[:freeze_count]
	trainable_groups = group_names[freeze_count:]

	for group_name in frozen_groups:
		for param in groups[group_name]:
			param.requires_grad = False

	logger.info(
		"Backbone freezing: freeze_up_to=%s, frozen_groups=%s, trainable_groups=%s",
		freeze_up_to,
		frozen_groups,
		trainable_groups,
	)

	if freeze_up_to > len(group_names):
		logger.warning(
			"Requested freeze_up_to=%s but backbone has only %s parameter groups",
			freeze_up_to,
			len(group_names),
		)

	return frozen_groups, trainable_groups


def load_backbone(
	name: str,
	pretrained: bool,
	freeze_up_to: int | None,
	model_cfg: Any | None = None,
) -> tuple[nn.Module, int]:
	"""Load a timm backbone as feature extractor and optionally freeze early groups.

	Args:
		name: timm model name.
		pretrained: Whether to load pretrained weights.
		freeze_up_to: Number of leading parameter groups to freeze.
		model_cfg: Model config with optional MixStyle settings.

	Returns:
		(backbone, feature_dim)
	"""
	logger = logging.getLogger(__name__)

	backbone = timm.create_model(
		name,
		pretrained=pretrained,
		features_only=False,
		num_classes=0,
	)
	feature_dim = _infer_feature_dim(backbone)

	_freeze_parameter_groups(backbone, freeze_up_to, logger)
	_attach_mixstyle_hooks(backbone, model_cfg=model_cfg, logger=logger)
	logger.info(
		"Backbone loaded: name=%s, pretrained=%s, feature_dim=%s",
		name,
		pretrained,
		feature_dim,
	)

	return backbone, feature_dim


__all__ = ["load_backbone"]


# Example usage and validation
if __name__ == "__main__":
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
	)

	model_name = "resnet18"
	backbone, feature_dim = load_backbone(
		name=model_name,
		pretrained=False,
		freeze_up_to=2,
		model_cfg=None,
	)

	assert isinstance(backbone, nn.Module), "Backbone must be an nn.Module"
	assert isinstance(feature_dim, int) and feature_dim > 0, "feature_dim must be a positive int"

	groups = _group_parameters(backbone)
	group_names = list(groups.keys())
	expected_frozen = min(2, len(group_names))

	frozen_group_names = group_names[:expected_frozen]
	trainable_group_names = group_names[expected_frozen:]

	for group_name in frozen_group_names:
		assert all(
			not param.requires_grad for param in groups[group_name]
		), f"Expected frozen group: {group_name}"

	for group_name in trainable_group_names:
		assert all(
			param.requires_grad for param in groups[group_name]
		), f"Expected trainable group: {group_name}"

	print(
		"Backbone loader check passed. "
		f"model={model_name}, feature_dim={feature_dim}, "
		f"frozen_groups={frozen_group_names}, trainable_groups={trainable_group_names}"
	)
