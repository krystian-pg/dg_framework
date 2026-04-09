from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PassThrough(nn.Module):
	"""Identity layer useful for toggling optional modules from config."""

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x


class DropBlock2d(nn.Module):
	"""DropBlock regularization for 2D feature maps."""

	def __init__(self, drop_prob: float = 0.1, block_size: int = 7) -> None:
		super().__init__()
		if not 0.0 <= drop_prob < 1.0:
			raise ValueError("drop_prob must be in [0, 1)")
		if block_size < 1:
			raise ValueError("block_size must be >= 1")
		self.drop_prob = drop_prob
		self.block_size = block_size

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.drop_prob == 0.0 or not self.training:
			return x
		if x.dim() != 4:
			raise ValueError("DropBlock2d expects input shape (N, C, H, W)")

		n, _, h, w = x.shape
		block = min(self.block_size, h, w)
		if block < 1:
			return x

		valid_h = h - block + 1
		valid_w = w - block + 1
		if valid_h <= 0 or valid_w <= 0:
			return x

		gamma = self.drop_prob * (h * w) / (block * block) / (valid_h * valid_w)
		gamma = min(max(gamma, 0.0), 1.0)

		mask = torch.bernoulli(
			torch.full((n, 1, h, w), gamma, device=x.device, dtype=x.dtype)
		)
		block_mask = F.max_pool2d(
			mask,
			kernel_size=block,
			stride=1,
			padding=block // 2,
		)
		if block % 2 == 0:
			block_mask = block_mask[:, :, :-1, :-1]

		keep_mask = 1.0 - block_mask
		keep_count = keep_mask.sum().clamp(min=1.0)
		scale = keep_mask.numel() / keep_count
		return x * keep_mask * scale


class SpectralNormWrapper(nn.Module):
	"""Wrap any module and apply spectral normalization to its weight parameter."""

	def __init__(
		self,
		module: nn.Module,
		name: str = "weight",
		n_power_iterations: int = 1,
		eps: float = 1e-12,
	) -> None:
		super().__init__()
		self.module = nn.utils.spectral_norm(
			module,
			name=name,
			n_power_iterations=n_power_iterations,
			eps=eps,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.module(x)


class DomainAgnosticBatchNorm2d(nn.Module):
	"""BatchNorm that always uses current-batch statistics (no running stats)."""

	def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
		super().__init__()
		self.bn = nn.BatchNorm2d(
			num_features=num_features,
			eps=eps,
			momentum=momentum,
			affine=True,
			track_running_stats=False,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.bn(x)


class BottleneckAdapter(nn.Module):
	"""Residual bottleneck adapter for lightweight feature injection."""

	def __init__(
		self,
		channels: int,
		reduction: int = 4,
		dropout: float = 0.0,
		activation: type[nn.Module] = nn.GELU,
	) -> None:
		super().__init__()
		if channels < 1:
			raise ValueError("channels must be >= 1")
		if reduction < 1:
			raise ValueError("reduction must be >= 1")
		if not 0.0 <= dropout < 1.0:
			raise ValueError("dropout must be in [0, 1)")

		hidden = max(1, channels // reduction)
		self.adapter = nn.Sequential(
			nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
			activation(),
			nn.Dropout2d(p=dropout),
			nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x + self.adapter(x)


__all__ = [
	"PassThrough",
	"DropBlock2d",
	"SpectralNormWrapper",
	"DomainAgnosticBatchNorm2d",
	"BottleneckAdapter",
]


# Example usage and validation
if __name__ == "__main__":
	torch.manual_seed(42)
	x = torch.randn(2, 64, 32, 32)

	identity = PassThrough()
	y_identity = identity(x)
	assert y_identity is x, "PassThrough should return the same tensor object"

	dropblock = DropBlock2d(drop_prob=0.2, block_size=5)
	dropblock.train()
	y_drop = dropblock(x)
	assert y_drop.shape == x.shape, "DropBlock2d must preserve input shape"
	assert torch.isfinite(y_drop).all(), "DropBlock2d output contains invalid values"

	conv_sn = SpectralNormWrapper(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))
	y_sn = conv_sn(x)
	assert y_sn.shape == x.shape, "SpectralNormWrapper must preserve expected conv output shape"
	assert hasattr(conv_sn.module, "weight_u"), "Spectral norm was not applied to wrapped module"

	dabn = DomainAgnosticBatchNorm2d(num_features=64)
	dabn.eval()
	y_bn = dabn(x)
	assert y_bn.shape == x.shape, "DomainAgnosticBatchNorm2d must preserve input shape"

	adapter = BottleneckAdapter(channels=64, reduction=4, dropout=0.1)
	y_adapter = adapter(x)
	assert y_adapter.shape == x.shape, "BottleneckAdapter must preserve input shape"

	print("Custom layers check passed: PassThrough, DropBlock2d, SpectralNormWrapper, DomainAgnosticBatchNorm2d, BottleneckAdapter")
