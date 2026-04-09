from __future__ import annotations

import logging
import sys
from pathlib import Path

try:
	from dg_framework.config import CFG
except ModuleNotFoundError:
	project_root = Path(__file__).resolve().parents[1]
	if str(project_root) not in sys.path:
		sys.path.append(str(project_root))
	from config import CFG


_SUMMARY_PRINTED = False


def _resolve_run_log(log_path: Path) -> Path:
	logs_dir = log_path if log_path.name == "logs" else log_path / "logs"
	logs_dir.mkdir(parents=True, exist_ok=True)
	return logs_dir / "run.log"


def _build_cfg_summary_lines() -> list[str]:
	rows = [
		("dataset", CFG.data.dataset_name),
		("target_domain", CFG.data.target_domain),
		("backbone", CFG.model.backbone_name),
		("epochs", str(CFG.train.epochs)),
		("batch_size", str(CFG.train.batch_size)),
		("lr", str(CFG.train.lr)),
		("seed", str(CFG.train.seed)),
		("experiment", CFG.experiment.name),
	]
	key_width = max(len(k) for k, _ in rows)
	val_width = max(len(v) for _, v in rows)
	border = "+-" + "-" * key_width + "-+-" + "-" * val_width + "-+"

	lines = ["CONFIG SUMMARY", border]
	for key, value in rows:
		lines.append(f"| {key:<{key_width}} | {value:<{val_width}} |")
	lines.append(border)
	return lines


def _print_summary_once(run_log: Path) -> None:
	global _SUMMARY_PRINTED
	if _SUMMARY_PRINTED:
		return

	lines = _build_cfg_summary_lines()
	for line in lines:
		print(line)

	with run_log.open("a", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")

	_SUMMARY_PRINTED = True


def get_logger(name: str, log_path: Path | str, level: str | int, print_global_summary: bool = False) -> logging.Logger:
	log_root = Path(log_path)
	run_log = _resolve_run_log(log_root)

	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)
	logger.propagate = False

	target_level = logging.getLevelName(level) if isinstance(level, str) else level
	if not isinstance(target_level, int):
		raise ValueError(f"Unsupported log level: {level}")

	if not logger.handlers:
		formatter = logging.Formatter(
			fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
			datefmt="%Y-%m-%d %H:%M:%S",
		)

		stream_handler = logging.StreamHandler(sys.stdout)
		stream_handler.setLevel(target_level)
		stream_handler.setFormatter(formatter)

		file_handler = logging.FileHandler(run_log, encoding="utf-8")
		file_handler.setLevel(target_level)
		file_handler.setFormatter(formatter)

		logger.addHandler(stream_handler)
		logger.addHandler(file_handler)
	else:
		for handler in logger.handlers:
			handler.setLevel(target_level)

	if print_global_summary:
		_print_summary_once(run_log)
	return logger


if __name__ == "__main__":
	test_root = Path(CFG.experiment.output_root) / CFG.experiment.name
	logger = get_logger("dg_pacs", test_root, "DEBUG", print_global_summary=True)

	logger.debug("debug message for logging check")
	logger.info("info message for logging check")
	logger.warning("warning message for logging check")

	assert any(
		isinstance(handler, logging.StreamHandler)
		and not isinstance(handler, logging.FileHandler)
		and getattr(handler, "stream", None) is sys.stdout
		for handler in logger.handlers
	), "Logger is missing stdout handler"

	run_log = _resolve_run_log(test_root)
	assert run_log.is_file(), "Missing logs/run.log"

	run_log_text = run_log.read_text(encoding="utf-8")
	assert "debug message for logging check" in run_log_text, "Missing debug line in run.log"
	assert "info message for logging check" in run_log_text, "Missing info line in run.log"
	assert "warning message for logging check" in run_log_text, "Missing warning line in run.log"

	print("Logging check passed. Verify console lines above for DEBUG/INFO/WARNING output.")
