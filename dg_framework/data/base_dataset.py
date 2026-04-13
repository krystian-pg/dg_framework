from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import DataLoader


class BaseDataset(ABC):
	@abstractmethod
	def get_domain_names(self) -> list[str]:
		...

	@abstractmethod
	def get_class_names(self) -> list[str]:
		...

	@abstractmethod
	def get_loaders(
		self,
		train_domains: list[str],
		test_domain: str,
		*args: Any,
		**kwargs: Any,
	) -> tuple[DataLoader, DataLoader, DataLoader]:
		...

	@property
	@abstractmethod
	def num_classes(self) -> int:
		...


if __name__ == "__main__":
	raised = False
	try:
		BaseDataset()
	except TypeError:
		raised = True

	assert raised, "BaseDataset should not be instantiable"
	print("ABC enforcement check passed.")
