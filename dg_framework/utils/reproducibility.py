from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
	"""Seed Python, NumPy, and PyTorch (CPU/CUDA) RNGs.

	When deterministic=True, this enables deterministic kernels for reproducibility.
	Note: deterministic execution can reduce training throughput.
	"""
	if deterministic:
		# Required by CuBLAS for deterministic CUDA behavior on CUDA >= 10.2.
		os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

	if deterministic:
		torch.use_deterministic_algorithms(True)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	else:
		torch.use_deterministic_algorithms(False)
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True

# Example usage and validation
if __name__ == "__main__":
	try:
		from dg_framework.config import CFG
	except ModuleNotFoundError:
		project_root = Path(__file__).resolve().parents[1]
		if str(project_root) not in sys.path:
			sys.path.append(str(project_root))
		from config import CFG

	set_seed(CFG.train.seed, deterministic=CFG.train.deterministic)
	tensor_a = torch.randn(4, 4)

	set_seed(CFG.train.seed, deterministic=CFG.train.deterministic)
	tensor_b = torch.randn(4, 4)

	assert torch.equal(tensor_a, tensor_b), "Seed reset reproducibility check failed"
	print("Reproducibility check passed.")
