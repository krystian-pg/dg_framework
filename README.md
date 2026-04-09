# DG Framework

Lightweight, modular framework for Domain Generalization experiments on PACS.

It provides a clear end-to-end training pipeline:

- reproducibility and experiment setup
- dataset loading and train/val/test split across domains
- backbone + classifier composition
- configurable losses, optimizer, scheduler, early stopping
- training loop with AMP, gradient clipping, gradient accumulation, and progress reporting
- evaluation with calibration-aware metrics, per-domain reports, confusion matrices, and optional embedding export

## Highlights

- Clean entry point with explicit execution order in `run.py`
- Device selection from config: `auto`, `cpu`, `cuda`, `mps` (Apple Silicon)
- Metric registry in one place (`evaluation/metrics.py`), including:
	- accuracy
	- f1_micro, f1_macro, f1_weighted
	- balanced_accuracy, cohen_kappa
	- expected_calibration_error (ECE)
	- brier_score
	- top_k_accuracy
- Per-domain evaluation for DG diagnostics
- Optional W&B logging with no-op behavior when disabled

## Repository Layout

```text
dg_framework/
	config.py
	run.py
	data/
		base_dataset.py
		pacs.py
		transforms.py
	models/
		backbone.py
		classifier.py
		layers.py
	training/
		early_stopping.py
		losses.py
		trainer.py
	evaluation/
		evaluator.py
		metrics.py
	utils/
		experiment.py
		logging_setup.py
		reproducibility.py
```

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- timm
- numpy
- pillow
- tqdm
- wandb (optional)

## Installation

From project root (where `dg_framework/` lives):

```bash
python3 -m venv ml
source ml/bin/activate
pip install --upgrade pip
pip install torch torchvision timm numpy pillow tqdm
# optional
pip install wandb
```

## Dataset Layout (PACS)

Default config expects:

```text
pacs_data/
	pacs_data/
		art_painting/
		cartoon/
		photo/
		sketch/
```

Each domain directory should contain class subdirectories with images.

Default path in config:

- `data.root_path = "pacs_data/pacs_data"`

## Quick Start

Run training from repository root:

```bash
python3 dg_framework/run.py
```

At startup, the framework prints:

- config summary
- dataset summary (split sizes, batches, domains, class distribution)
- model block diagram
- runtime summary (selected device, parameter counts, effective batch size, etc.)

## Configuration

Main settings are in `config.py`.

### Device Selection

Set in `TrainConfig.device`:

- `"auto"`: prefer CUDA, else MPS, else CPU
- `"cuda"`: require CUDA (error if unavailable)
- `"mps"`: require Apple MPS (error if unavailable)
- `"cpu"`: force CPU

### Training Progress

- `train.show_progress = True` enables progress output
- if terminal supports TTY and tqdm is installed: stable progress bars
- otherwise: periodic step logs
- `train.progress_log_interval` controls fallback logging frequency

### Useful Training Knobs

- `train.grad_accum_steps`
- `train.grad_clip`
- `train.amp`
- `train.optimiser` (`adamw` or `sgd`)
- `train.scheduler.scheduler_type` (`none`, `cosine`, `step`, `plateau`)
- `train.lr_warmup_epochs`

### Evaluation Knobs

- `evaluation.metrics`
- `evaluation.per_domain_eval`
- `evaluation.tta.enabled`, `evaluation.tta.n_augments`
- `experiment.save_confusion_matrix`
- `experiment.export_embeddings`

## Outputs

For experiment name `experiment.name`, outputs are created under:

```text
<experiment.output_root>/<experiment.name>/
	config.py
	results.json
	checkpoints/
		best.pt
	logs/
		run.log
	figures/
		confusion_matrix.png
		confusion_matrix_<domain>.png
	embeddings/
		eval_embeddings.npz
		eval_embeddings_<domain>.npz
```

`results.json` contains:

- train/val history
- test metrics
- per-domain metrics (if enabled)
- selected device
- best checkpoint path

## Run Modes

Default:

```bash
python3 dg_framework/run.py
```

Integration smoke test mode:

```bash
DG_RUN_MODE=integration python3 dg_framework/run.py
```

## Extending the Framework

- Add a new loss: register it in `training/losses.py`
- Add a new metric: register it in `evaluation/metrics.py`
- Swap backbone: set `model.backbone_name` (timm model)
- Modify classifier head: use `model.head_type`, `model.head_depth`, `model.head_width`

## Troubleshooting

### It still runs on CPU

- verify `train.device` in `config.py`
- `mps`/`cuda` modes are strict and should raise if unavailable
- check startup logs for:
	- requested device
	- selected device
	- backend availability

### Progress bars look broken

- use a regular terminal (TTY)
- ensure `tqdm` is installed
- if no TTY, framework falls back to periodic logs (expected)

### Missing dependency: timm

Install with:

```bash
pip install timm
```