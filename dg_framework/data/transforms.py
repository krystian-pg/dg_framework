from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize_size(img_size: int | tuple[int, int]) -> int:
	if isinstance(img_size, tuple):
		if len(img_size) != 2 or img_size[0] != img_size[1]:
			raise ValueError("img_size tuple must be square, e.g. (224, 224)")
		img_size = img_size[0]

	if img_size < 1:
		raise ValueError("img_size must be >= 1")

	return img_size


def get_train_transform(
	img_size: int | tuple[int, int],
	mean: Sequence[float] = IMAGENET_MEAN,
	std: Sequence[float] = IMAGENET_STD,
	hflip_prob: float = 0.5,
	color_jitter_strength: float = 0.2,
) -> transforms.Compose:
	size = _normalize_size(img_size)
	hue_strength = max(0.0, min(0.1, color_jitter_strength / 2))
	return transforms.Compose(
		[
			transforms.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
			transforms.RandomHorizontalFlip(p=hflip_prob),
			transforms.ToTensor(),
			transforms.ColorJitter(
				brightness=color_jitter_strength,
				contrast=color_jitter_strength,
				saturation=color_jitter_strength,
				hue=hue_strength,
			),
			transforms.Normalize(mean=mean, std=std),
		]
	)


def get_eval_transform(
	img_size: int | tuple[int, int],
	mean: Sequence[float] = IMAGENET_MEAN,
	std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
	size = _normalize_size(img_size)
	return transforms.Compose(
		[
			transforms.Resize((size, size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]
	)


def get_tta_transforms(
	img_size: int | tuple[int, int],
	n_augments: int,
	mean: Sequence[float] = IMAGENET_MEAN,
	std: Sequence[float] = IMAGENET_STD,
	hflip_prob: float = 0.5,
	color_jitter_strength: float = 0.1,
) -> list[transforms.Compose]:
	size = _normalize_size(img_size)
	if n_augments < 1:
		raise ValueError("n_augments must be >= 1")
	hue_strength = max(0.0, min(0.05, color_jitter_strength / 2))

	tta_list: list[transforms.Compose] = []
	for _ in range(n_augments):
		tta_list.append(
			transforms.Compose(
				[
					transforms.RandomResizedCrop(size=size, scale=(0.9, 1.0)),
					transforms.RandomHorizontalFlip(p=hflip_prob),
					# Run jitter on tensors to avoid PIL hue overflow with older torchvision+numpy combos.
					transforms.ToTensor(),
					transforms.ColorJitter(
						brightness=color_jitter_strength,
						contrast=color_jitter_strength,
						saturation=color_jitter_strength,
						hue=hue_strength,
					),
					transforms.Normalize(mean=mean, std=std),
				]
			)
		)
	return tta_list

# Example usage and validation
if __name__ == "__main__":
	img_size = 224
	random_image = np.random.randint(0, 256, size=(320, 320, 3), dtype=np.uint8)
	pil_image = Image.fromarray(random_image)

	train_transform = get_train_transform(img_size)
	eval_transform = get_eval_transform(img_size)
	tta_transforms = get_tta_transforms(img_size, n_augments=4)

	train_out = train_transform(pil_image)
	eval_out = eval_transform(pil_image)

	assert tuple(train_out.shape) == (3, img_size, img_size), "Train transform shape mismatch"
	assert tuple(eval_out.shape) == (3, img_size, img_size), "Eval transform shape mismatch"

	for i, tta_transform in enumerate(tta_transforms):
		out = tta_transform(pil_image)
		assert tuple(out.shape) == (3, img_size, img_size), f"TTA[{i}] shape mismatch"

	print("Transform checks passed.")
