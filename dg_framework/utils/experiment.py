from __future__ import annotations

import shutil
import sys
from pathlib import Path

try:
	from dg_framework.config import CFG, Config
except ModuleNotFoundError:
	project_root = Path(__file__).resolve().parents[1]
	if str(project_root) not in sys.path:
		sys.path.append(str(project_root))
	from config import CFG, Config


def setup_experiment(cfg: Config) -> Path:
	experiment_path = Path(cfg.experiment.output_root) / cfg.experiment.name
	experiment_path.mkdir(parents=True, exist_ok=True)

	for subdir in ("checkpoints", "logs", "figures", "embeddings"):
		(experiment_path / subdir).mkdir(parents=True, exist_ok=True)

	source_config = Path(__file__).resolve().parents[1] / "config.py"
	target_config = experiment_path / "config.py"
	shutil.copy2(source_config, target_config)

	return experiment_path

# Example usage and validation
if __name__ == "__main__":
	exp_path = setup_experiment(CFG)

	for subdir in ("checkpoints", "logs", "figures", "embeddings"):
		assert (exp_path / subdir).is_dir(), f"Missing subdirectory: {subdir}"

	assert (exp_path / "config.py").is_file(), "Missing copied config.py"
	print(f"Experiment setup check passed at: {exp_path}")
