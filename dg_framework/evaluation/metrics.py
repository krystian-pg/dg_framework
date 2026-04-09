from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _validate_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	y_true_arr = np.asarray(y_true)
	y_prob_arr = np.asarray(y_prob)

	if y_true_arr.ndim != 1:
		raise ValueError("y_true must be a 1D array of class indices")
	if y_prob_arr.ndim != 2:
		raise ValueError("y_prob must be a 2D array of shape (n_samples, n_classes)")
	if y_true_arr.shape[0] != y_prob_arr.shape[0]:
		raise ValueError("y_true and y_prob must have the same number of samples")
	if y_prob_arr.shape[1] < 2:
		raise ValueError("y_prob must contain probabilities for at least 2 classes")

	y_true_arr = y_true_arr.astype(np.int64, copy=False)
	if np.any(y_true_arr < 0) or np.any(y_true_arr >= y_prob_arr.shape[1]):
		raise ValueError("y_true contains class indices outside [0, n_classes)")

	# Normalize probabilities if needed to make metrics robust to minor numeric drift.
	row_sums = y_prob_arr.sum(axis=1, keepdims=True)
	if np.any(row_sums <= 0):
		raise ValueError("Each row in y_prob must have a positive sum")
	y_prob_arr = y_prob_arr / row_sums

	return y_true_arr, y_prob_arr


def _hard_predictions(y_prob: np.ndarray) -> np.ndarray:
	return np.argmax(y_prob, axis=1)


def _per_class_stats(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	tp = np.zeros(n_classes, dtype=np.float64)
	fp = np.zeros(n_classes, dtype=np.float64)
	fn = np.zeros(n_classes, dtype=np.float64)

	for cls in range(n_classes):
		tp[cls] = np.sum((y_true == cls) & (y_pred == cls))
		fp[cls] = np.sum((y_true != cls) & (y_pred == cls))
		fn[cls] = np.sum((y_true == cls) & (y_pred != cls))

	return tp, fp, fn


def accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	y_true, y_prob = _validate_inputs(y_true, y_prob)
	y_pred = _hard_predictions(y_prob)
	return float(np.mean(y_pred == y_true))


def f1_micro(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	y_true, y_prob = _validate_inputs(y_true, y_prob)
	y_pred = _hard_predictions(y_prob)
	n_classes = y_prob.shape[1]
	tp, fp, fn = _per_class_stats(y_true, y_pred, n_classes)

	tp_sum = np.sum(tp)
	fp_sum = np.sum(fp)
	fn_sum = np.sum(fn)
	precision = tp_sum / (tp_sum + fp_sum + 1e-12)
	recall = tp_sum / (tp_sum + fn_sum + 1e-12)
	f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
	return float(f1)


def f1_macro(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	y_true, y_prob = _validate_inputs(y_true, y_prob)
	y_pred = _hard_predictions(y_prob)
	n_classes = y_prob.shape[1]
	tp, fp, fn = _per_class_stats(y_true, y_pred, n_classes)

	precision = tp / (tp + fp + 1e-12)
	recall = tp / (tp + fn + 1e-12)
	f1_per_class = 2.0 * precision * recall / (precision + recall + 1e-12)
	return float(np.mean(f1_per_class))


def f1_weighted(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	y_true, y_prob = _validate_inputs(y_true, y_prob)
	y_pred = _hard_predictions(y_prob)
	n_classes = y_prob.shape[1]
	tp, fp, fn = _per_class_stats(y_true, y_pred, n_classes)

	precision = tp / (tp + fp + 1e-12)
	recall = tp / (tp + fn + 1e-12)
	f1_per_class = 2.0 * precision * recall / (precision + recall + 1e-12)
	support = np.bincount(y_true, minlength=n_classes).astype(np.float64)
	if support.sum() <= 0:
		return 0.0
	weights = support / support.sum()
	return float(np.sum(f1_per_class * weights))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	y_true, y_prob = _validate_inputs(y_true, y_prob)
	n_samples, n_classes = y_prob.shape
	y_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
	y_onehot[np.arange(n_samples), y_true] = 1.0

	# Multiclass Brier score: average squared error over classes and samples.
	score = np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1))
	return float(score)


def balanced_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	y_true, y_prob = _validate_inputs(y_true, y_prob)
	y_pred = _hard_predictions(y_prob)
	n_classes = y_prob.shape[1]

	recall_per_class: list[float] = []
	for cls in range(n_classes):
		positives = np.sum(y_true == cls)
		if positives == 0:
			continue
		tp = np.sum((y_true == cls) & (y_pred == cls))
		recall_per_class.append(tp / positives)

	if not recall_per_class:
		return 0.0
	return float(np.mean(recall_per_class))


def cohen_kappa(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	y_true, y_prob = _validate_inputs(y_true, y_prob)
	y_pred = _hard_predictions(y_prob)
	n_classes = y_prob.shape[1]

	conf_mat = np.zeros((n_classes, n_classes), dtype=np.float64)
	for t, p in zip(y_true, y_pred, strict=True):
		conf_mat[t, p] += 1.0

	total = conf_mat.sum()
	if total <= 0:
		return 0.0

	po = np.trace(conf_mat) / total
	row_marginals = conf_mat.sum(axis=1)
	col_marginals = conf_mat.sum(axis=0)
	pe = np.sum(row_marginals * col_marginals) / (total ** 2)

	den = 1.0 - pe
	if abs(den) < 1e-12:
		return 0.0
	return float((po - pe) / den)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	y_true, y_prob = _validate_inputs(y_true, y_prob)
	confidences = np.max(y_prob, axis=1)
	predictions = np.argmax(y_prob, axis=1)
	accuracies = (predictions == y_true).astype(np.float64)

	n_bins = 15
	bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
	ece = 0.0

	for i in range(n_bins):
		low = bin_edges[i]
		high = bin_edges[i + 1]
		in_bin = (confidences > low) & (confidences <= high)
		bin_size = np.sum(in_bin)
		if bin_size == 0:
			continue

		bin_acc = np.mean(accuracies[in_bin])
		bin_conf = np.mean(confidences[in_bin])
		ece += abs(bin_acc - bin_conf) * (bin_size / len(y_true))

	return float(ece)


def top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	y_true, y_prob = _validate_inputs(y_true, y_prob)
	k = min(5, y_prob.shape[1])
	top_k = np.argpartition(y_prob, -k, axis=1)[:, -k:]
	correct = np.any(top_k == y_true[:, None], axis=1)
	return float(np.mean(correct))


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
	"accuracy": accuracy,
	"f1_micro": f1_micro,
	"f1_macro": f1_macro,
	"f1_weighted": f1_weighted,
	"brier_score": brier_score,
	"balanced_accuracy": balanced_accuracy,
	"cohen_kappa": cohen_kappa,
	"expected_calibration_error": expected_calibration_error,
	"ece": expected_calibration_error,
	"top_k_accuracy": top_k_accuracy,
}


def compute_all(metrics: list[str], y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
	results: dict[str, float] = {}
	for metric_name in metrics:
		if metric_name not in METRICS:
			available = ", ".join(sorted(METRICS.keys()))
			raise KeyError(f"Unknown metric '{metric_name}'. Available: {available}")
		results[metric_name] = METRICS[metric_name](y_true, y_prob)
	return results


__all__ = ["METRICS", "compute_all"]


# Example usage and validation
if __name__ == "__main__":
	rng = np.random.default_rng(42)
	n_samples = 24
	n_classes = 7

	y_true_demo = rng.integers(0, n_classes, size=n_samples, dtype=np.int64)
	raw_scores = rng.random((n_samples, n_classes))
	y_prob_demo = raw_scores / raw_scores.sum(axis=1, keepdims=True)

	required_metrics = [
		"accuracy",
		"f1_micro",
		"f1_macro",
		"f1_weighted",
		"brier_score",
		"balanced_accuracy",
		"cohen_kappa",
		"expected_calibration_error",
		"top_k_accuracy",
	]

	for metric_name in required_metrics:
		assert metric_name in METRICS, f"Missing required metric in registry: {metric_name}"

	results = compute_all(required_metrics, y_true_demo, y_prob_demo)

	assert set(results.keys()) == set(required_metrics), "compute_all returned unexpected metric keys"

	for metric_name, value in results.items():
		assert np.isfinite(value), f"Metric {metric_name} returned non-finite value"

	for bounded_metric in ["accuracy", "f1_micro", "f1_macro", "f1_weighted", "balanced_accuracy", "top_k_accuracy"]:
		metric_value = results[bounded_metric]
		assert 0.0 <= metric_value <= 1.0, f"{bounded_metric} should be in [0, 1], got: {metric_value}"

	ece_alias_value = METRICS["ece"](y_true_demo, y_prob_demo)
	ece_full_name_value = METRICS["expected_calibration_error"](y_true_demo, y_prob_demo)
	assert np.isclose(ece_alias_value, ece_full_name_value), "ECE alias mismatch"

	print(
		"Metrics registry check passed. "
		f"computed={sorted(results.keys())}, "
		f"accuracy={results['accuracy']:.4f}, "
		f"brier={results['brier_score']:.4f}, "
		f"ece={results['expected_calibration_error']:.4f}"
	)
