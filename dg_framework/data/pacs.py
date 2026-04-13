from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
	from dg_framework.config import CFG, Config
	from dg_framework.data.base_dataset import BaseDataset
	from dg_framework.data.transforms import get_eval_transform, get_train_transform
except ModuleNotFoundError:
	project_root = Path(__file__).resolve().parents[1]
	if str(project_root) not in sys.path:
		sys.path.append(str(project_root))
	from config import CFG, Config
	from data.base_dataset import BaseDataset
	from data.transforms import get_eval_transform, get_train_transform


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class PACSSample:
	path: Path
	label: int
	domain_id: int


class _PACSTorchDataset(Dataset[tuple[Tensor, int, int, str]]):
	def __init__(self, samples: list[PACSSample], transform: Any) -> None:
		self.samples = samples
		self.transform = transform

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int) -> tuple[Tensor, int, int, str]:
		sample = self.samples[index]
		with Image.open(sample.path) as img:
			image = img.convert("RGB")
		image_tensor = self.transform(image)
		return image_tensor, sample.label, sample.domain_id, str(sample.path)


class PACSDataset(BaseDataset):
	def __init__(self, cfg: Config) -> None:
		self.cfg = cfg
		self.root_path = self._resolve_root_path(cfg.data.root_path)
		self._domain_names = self._discover_domains()
		self._class_names = self._discover_classes()
		self._class_to_idx = {name: idx for idx, name in enumerate(self._class_names)}
		self._domain_to_id = {name: idx for idx, name in enumerate(self._domain_names)}
		self._samples_by_domain = self._build_samples_by_domain()

	def get_domain_names(self) -> list[str]:
		return list(self._domain_names)

	def get_class_names(self) -> list[str]:
		return list(self._class_names)

	@property
	def num_classes(self) -> int:
		return len(self._class_names)

	def get_loaders(
		self,
		train_domains: list[str],
		test_domain: str,
		batch_size: int | None = None,
		num_workers: int | None = None,
		pin_memory: bool = True,
		drop_last: bool = False,
		*args: Any,
		**kwargs: Any,
	) -> tuple[DataLoader, DataLoader, DataLoader]:
		del args, kwargs

		if test_domain not in self._domain_to_id:
			raise ValueError(f"Unknown test domain: {test_domain}")

		if not train_domains:
			raise ValueError("train_domains must contain at least one domain")

		invalid_domains = [d for d in train_domains if d not in self._domain_to_id]
		if invalid_domains:
			raise ValueError(f"Unknown train domains: {invalid_domains}")

		if test_domain in train_domains:
			raise ValueError("test_domain cannot appear in train_domains")

		train_pool: list[PACSSample] = []
		for domain in train_domains:
			train_pool.extend(self._samples_by_domain[domain])

		if not train_pool:
			raise ValueError("No training samples found for selected train domains")

		test_samples = list(self._samples_by_domain[test_domain])
		if not test_samples:
			raise ValueError(f"No test samples found for domain: {test_domain}")

		train_samples, val_samples = self._stratified_train_val_split(
			train_pool,
			train_ratio=self.cfg.data.train_val_split,
			seed=self.cfg.train.seed,
		)

		bs = batch_size or self.cfg.train.batch_size
		nw = self.cfg.data.num_workers if num_workers is None else num_workers

		train_dataset = _PACSTorchDataset(train_samples, get_train_transform(self.cfg.data.image_size))
		eval_transform = get_eval_transform(self.cfg.data.image_size)
		val_dataset = _PACSTorchDataset(val_samples, eval_transform)
		test_dataset = _PACSTorchDataset(test_samples, eval_transform)

		train_loader = DataLoader(
			train_dataset,
			batch_size=bs,
			shuffle=True,
			num_workers=nw,
			pin_memory=pin_memory,
			drop_last=drop_last,
		)
		val_loader = DataLoader(
			val_dataset,
			batch_size=bs,
			shuffle=False,
			num_workers=nw,
			pin_memory=pin_memory,
			drop_last=False,
		)
		test_loader = DataLoader(
			test_dataset,
			batch_size=bs,
			shuffle=False,
			num_workers=nw,
			pin_memory=pin_memory,
			drop_last=False,
		)

		return train_loader, val_loader, test_loader

	def _resolve_root_path(self, root_path: str) -> Path:
		candidate = Path(root_path)
		workspace_root = Path(__file__).resolve().parents[2]
		search_paths = [candidate, Path.cwd() / candidate, workspace_root / candidate]

		for path in search_paths:
			if path.exists() and path.is_dir():
				return path.resolve()

		raise FileNotFoundError(
			f"Could not resolve dataset root: {root_path}. Checked: {[str(p) for p in search_paths]}"
		)

	def _discover_domains(self) -> list[str]:
		domains = sorted([p.name for p in self.root_path.iterdir() if p.is_dir()])
		if not domains:
			raise ValueError(f"No domains found in dataset root: {self.root_path}")
		return domains

	def _discover_classes(self) -> list[str]:
		class_names: set[str] = set()
		for domain in self._domain_names:
			domain_dir = self.root_path / domain
			for class_dir in domain_dir.iterdir():
				if class_dir.is_dir():
					class_names.add(class_dir.name)

		if not class_names:
			raise ValueError(f"No class folders found under: {self.root_path}")

		return sorted(class_names)

	def _build_samples_by_domain(self) -> dict[str, list[PACSSample]]:
		samples_by_domain: dict[str, list[PACSSample]] = {}

		for domain in self._domain_names:
			domain_id = self._domain_to_id[domain]
			domain_samples: list[PACSSample] = []

			for class_name in self._class_names:
				class_dir = self.root_path / domain / class_name
				if not class_dir.exists() or not class_dir.is_dir():
					continue

				label = self._class_to_idx[class_name]
				for image_path in sorted(class_dir.iterdir()):
					if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
						domain_samples.append(
							PACSSample(path=image_path, label=label, domain_id=domain_id)
						)

			if not domain_samples:
				raise ValueError(f"No image samples found for domain: {domain}")

			samples_by_domain[domain] = domain_samples

		return samples_by_domain

	@staticmethod
	def _stratified_train_val_split(
		samples: list[PACSSample],
		train_ratio: float,
		seed: int,
	) -> tuple[list[PACSSample], list[PACSSample]]:
		if not 0.0 < train_ratio < 1.0:
			raise ValueError("train_ratio must be in (0, 1)")

		buckets: dict[int, list[PACSSample]] = defaultdict(list)
		for sample in samples:
			buckets[sample.label].append(sample)

		rng = random.Random(seed)
		train_samples: list[PACSSample] = []
		val_samples: list[PACSSample] = []

		for _, class_samples in sorted(buckets.items(), key=lambda item: item[0]):
			rng.shuffle(class_samples)
			if len(class_samples) == 1:
				n_train = 1
			else:
				n_train = int(round(len(class_samples) * train_ratio))
				n_train = max(1, min(len(class_samples) - 1, n_train))

			train_samples.extend(class_samples[:n_train])
			val_samples.extend(class_samples[n_train:])

		if not val_samples:
			moved = train_samples.pop()
			val_samples.append(moved)

		rng.shuffle(train_samples)
		rng.shuffle(val_samples)
		return train_samples, val_samples


class PACS(PACSDataset):
	pass


def _print_batch_stats(split_name: str, loader: DataLoader) -> None:
	images, labels, domain_ids, _ = next(iter(loader))
	label_dist = dict(sorted(Counter(labels.tolist()).items()))
	domain_values = sorted(set(domain_ids.tolist()))

	print(f"{split_name} images shape: {tuple(images.shape)}")
	print(f"{split_name} labels shape: {tuple(labels.shape)}")
	print(f"{split_name} label distribution: {label_dist}")
	print(f"{split_name} domain IDs in batch: {domain_values}")


if __name__ == "__main__":
	dataset = PACS(CFG)
	test_domain = CFG.data.target_domain
	train_domains = [d for d in dataset.get_domain_names() if d != test_domain]

	train_loader, val_loader, test_loader = dataset.get_loaders(
		train_domains=train_domains,
		test_domain=test_domain,
	)

	_print_batch_stats("train", train_loader)
	_print_batch_stats("val", val_loader)
	_print_batch_stats("test", test_loader)

	train_domain_ids = {sample.domain_id for sample in train_loader.dataset.samples}
	test_domain_ids = {sample.domain_id for sample in test_loader.dataset.samples}

	assert train_domain_ids.isdisjoint(test_domain_ids), "Train and test domains overlap"
	print("Domain overlap check passed.")
