"""
Federated holdout test for MEDfl real-world pipelines.

After the last round, every client tests the final global model on its own
local holdout CSV (a file never used for training). Clients only send
aggregate numbers back to the server:

    - the confusion matrix at the decision threshold,
    - their local metrics,
    - two histograms of the predicted probabilities (positives / negatives).

The server sums the confusion matrices to get exact pooled metrics
(sensitivity, specificity, PPV, NPV, F1, accuracy) and sums the histograms
to rebuild a pooled ROC curve and AUC. The pooled AUC is exact up to the
histogram bin width: two scores falling in the same bin count as a tie.

The holdout goes through the same preprocessing as the client training
data: the server features in the server order, the raw values (the
real-world clients do not impute or scale), and a binary 0/1 target.

Optimized threshold (threshold_source="optimized"): in the same final
round, every client also sends the two score histograms of its whole
learning set (train + validation + test splits, LEARNING_SET). The server
sums them and picks the threshold among the histogram bin edges (Youden's
J or a target sensitivity). The training rows were used to fit the model,
so the learning-set metrics at that threshold are optimistic: the holdout
gives the unbiased estimate. At a bin edge, the histograms give exact confusion matrices,
so the holdout results are then recomputed from the holdout histograms
at that threshold. The holdout histograms are never used to choose it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Statuses a client can report in metrics["holdout_status"]
HOLDOUT_COMPLETED = "completed"
HOLDOUT_SKIPPED = "skipped"  # fewer rows than minimum_samples
HOLDOUT_MISSING = "missing"  # no holdout file on this client
HOLDOUT_FAILED = "failed"
HOLDOUT_UNSUPPORTED = "unsupported_task"

THRESHOLD_SOURCES = ("manual", "optimized")
# data the clients use for the optimized threshold (sent back in their summary, so a client running an
# older MEDfl that only sends its validation split is detected)
LEARNING_SET = "learning_set"
LEARNING_SET_SPLITS = ("train", "validation", "test")
# criterion key -> label shown in the results
THRESHOLD_CRITERIA = {"youden": "Youden's J", "target_sensitivity": "Target sensitivity"}


@dataclass
class HoldoutConfig:
    """
    Configuration for the optional post-training federated holdout test.
    """

    enabled: bool = False

    # A prediction is positive when its probability is >= threshold
    threshold: float = 0.5

    # Number of probability bins used to rebuild the pooled ROC curve / AUC
    num_bins: int = 100

    # Clients with fewer holdout rows do not send results
    minimum_samples: int = 10

    include_client_results: bool = True

    # "manual": the threshold above. "optimized": chosen on the clients'
    # learning sets (train + validation + test), at a histogram bin edge
    # (resolution 1 / num_bins)
    threshold_source: str = "manual"

    # "youden" or "target_sensitivity" (optimized threshold only)
    threshold_criterion: str = "youden"

    # minimal pooled learning-set sensitivity of the target_sensitivity criterion
    target_sensitivity: float = 0.9

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("Holdout enabled must be a boolean")

        if not 0.0 < float(self.threshold) < 1.0:
            raise ValueError("Holdout threshold must be between 0 and 1")

        if int(self.num_bins) < 2:
            raise ValueError("Holdout num_bins must be at least 2")

        if int(self.minimum_samples) <= 0:
            raise ValueError("Holdout minimum_samples must be greater than zero")

        if self.threshold_source not in THRESHOLD_SOURCES:
            raise ValueError(
                f"Holdout threshold_source must be one of {THRESHOLD_SOURCES}"
            )

        if self.threshold_criterion not in THRESHOLD_CRITERIA:
            raise ValueError(
                f"Holdout threshold_criterion must be one of {tuple(THRESHOLD_CRITERIA)}"
            )

        if not 0.0 < float(self.target_sensitivity) <= 1.0:
            raise ValueError("Holdout target_sensitivity must be in (0, 1]")

    def to_evaluate_config(self) -> Dict[str, Any]:
        """
        Keys added to the evaluate config of the final round.
        Flower configs are Scalar-only.
        """
        return {
            "holdout_enabled": True,
            "holdout_threshold": float(self.threshold),
            "holdout_num_bins": int(self.num_bins),
            "holdout_minimum_samples": int(self.minimum_samples),
            # the clients also send their learning-set histograms
            "threshold_optimization_enabled": self.threshold_source == "optimized",
        }


# ----------------------------------------------------------------------
# Client side
# ----------------------------------------------------------------------


def load_holdout(
    path: str,
    features: List[str],
    target: str,
    drop_missing_features: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Read a holdout CSV with the training schema.

    Rows without a target are dropped. Rows with a missing feature value are
    dropped when drop_missing_features is True (the neural network cannot
    predict them); XGBoost keeps them since it handles missing values.

    Returns (X, y, dataset information).
    """
    data = pd.read_csv(path)

    if target not in data.columns:
        raise ValueError(
            f"The holdout dataset has no '{target}' column (the model target)"
        )

    missing = [column for column in features if column not in data.columns]
    if missing:
        raise ValueError(
            f"The holdout dataset is missing {len(missing)} training "
            f"column(s): {', '.join(missing)}"
        )

    X = data[features].apply(pd.to_numeric, errors="coerce")
    non_numeric = [
        column
        for column in features
        if data[column].notna().any() and X[column].isna().all()
    ]
    if non_numeric:
        raise ValueError(
            f"These holdout columns are not numeric: {', '.join(non_numeric)}"
        )

    y = pd.to_numeric(data[target], errors="coerce")
    has_target = y.notna().values
    has_features = X.notna().all(axis=1).values

    keep = has_target & has_features if drop_missing_features else has_target
    X, y = X.values[keep], y.values[keep].astype(float)

    classes = set(np.unique(y).tolist())
    if not classes.issubset({0.0, 1.0}):
        raise ValueError(
            f"The target '{target}' must be binary (0/1) in the holdout "
            f"dataset, found {len(classes)} values: {sorted(classes)[:10]}"
        )

    info = {
        "name": os.path.basename(path),
        "rows_in_file": int(len(data)),
        "rows_without_target": int((~has_target).sum()),
        "rows_with_missing_features": int(
            (has_target & ~has_features).sum()
        ),
        "rows_missing_features_dropped": bool(drop_missing_features),
    }
    return X.astype(np.float32), y, info


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    """None when undefined (e.g. no positive case), so JSON stays valid."""
    return float(numerator) / float(denominator) if denominator else None


def _confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, int]:
    y_pred = (y_prob >= threshold).astype(float)
    return {
        "TP": int(((y_pred == 1) & (y_true == 1)).sum()),
        "FP": int(((y_pred == 1) & (y_true == 0)).sum()),
        "FN": int(((y_pred == 0) & (y_true == 1)).sum()),
        "TN": int(((y_pred == 0) & (y_true == 0)).sum()),
    }


def confusion_metrics(confusion: Dict[str, int]) -> Dict[str, Optional[float]]:
    """Threshold metrics from a confusion matrix (same definitions as the simulation holdout)."""
    tp, fp, fn, tn = (confusion[key] for key in ("TP", "FP", "FN", "TN"))
    return {
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        # same value as 2 * sensitivity * ppv / (sensitivity + ppv), and 0 when there is no true positive
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def _bin_edges(num_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, int(num_bins) + 1)


def local_holdout_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    num_bins: int,
    minimum_samples: int,
    client_id: str,
    dataset: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summary of one client's holdout test, safe to send to the server.
    Below minimum_samples rows, only the row count is sent.
    """
    rows = int(len(y_true))
    positives = int(y_true.sum())
    summary = {
        "client_id": client_id,
        "dataset": {**dataset, "rows": rows, "positives": positives, "negatives": rows - positives},
    }

    if rows < int(minimum_samples):
        return {
            **summary,
            "status": HOLDOUT_SKIPPED,
            "reason": (
                f"{rows} usable holdout row(s), fewer than the minimum of "
                f"{int(minimum_samples)}"
            ),
        }

    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    confusion = _confusion(y_true, y_prob, threshold)
    edges = _bin_edges(num_bins)

    metrics = {
        "auc": (
            float(roc_auc_score(y_true, y_prob))
            if 0 < positives < rows
            else None
        ),
        **confusion_metrics(confusion),
    }

    return {
        **summary,
        "status": HOLDOUT_COMPLETED,
        "threshold": float(threshold),
        "num_bins": int(num_bins),
        "metrics": metrics,
        "confusion_matrix": confusion,
        "pos_hist": np.histogram(y_prob[y_true == 1], bins=edges)[0].astype(int).tolist(),
        "neg_hist": np.histogram(y_prob[y_true == 0], bins=edges)[0].astype(int).tolist(),
    }


def local_validation_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int,
    minimum_samples: int,
    client_id: str,
    splits: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Summary of one client's learning set (train + validation + test) for the
    optimized threshold, safe to send to the server: counts and the two score
    histograms (same bins as the holdout). Below minimum_samples rows, only
    the counts are sent. splits gives the number of rows of each split.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)

    # e.g. missing feature values in the neural network input
    predicted = np.isfinite(y_prob)
    y_true, y_prob = y_true[predicted], y_prob[predicted]

    classes = set(np.unique(y_true).tolist())
    if not classes.issubset({0.0, 1.0}):
        raise ValueError(
            f"The learning-set target must be binary (0/1), found {len(classes)} "
            f"values: {sorted(classes)[:10]}"
        )

    rows = int(len(y_true))
    positives = int(y_true.sum())
    summary = {
        "client_id": client_id,
        "data": LEARNING_SET,
        "dataset": {
            "rows": rows,
            "positives": positives,
            "negatives": rows - positives,
            "rows_without_prediction": int((~predicted).sum()),
            "splits": {key: int(value) for key, value in (splits or {}).items()},
        },
    }

    if rows == 0:
        return {
            **summary,
            "status": HOLDOUT_MISSING,
            "reason": "No learning-set rows",
        }

    if rows < int(minimum_samples):
        return {
            **summary,
            "status": HOLDOUT_SKIPPED,
            "reason": (
                f"{rows} learning-set row(s), fewer than the minimum of "
                f"{int(minimum_samples)}"
            ),
        }

    y_prob = np.clip(y_prob, 0.0, 1.0)
    edges = _bin_edges(num_bins)
    return {
        **summary,
        "status": HOLDOUT_COMPLETED,
        "num_bins": int(num_bins),
        "pos_hist": np.histogram(y_prob[y_true == 1], bins=edges)[0].astype(int).tolist(),
        "neg_hist": np.histogram(y_prob[y_true == 0], bins=edges)[0].astype(int).tolist(),
    }


def run_local_validation(
    config: Dict[str, Any],
    validation: Optional[Callable[[], Tuple[np.ndarray, np.ndarray, Dict[str, int]]]],
    client_id: str,
) -> Dict[str, Any]:
    """
    Entries to add to the Flower evaluate metrics for the optimized
    threshold: threshold_validation_status, then threshold_validation
    (JSON) or threshold_validation_error. (The metric names are kept from
    the validation-only versions; the data is the learning set.)

    validation returns (y_true, y_prob, splits) of the client's whole
    learning set (train + validation + test) with the final global model,
    splits being the number of rows of each split.
    """
    if validation is None:
        return {
            "threshold_validation_status": HOLDOUT_MISSING,
            "threshold_validation_error": "This client cannot send learning-set data",
        }

    try:
        y_true, y_prob, splits = validation()
        summary = local_validation_summary(
            y_true=y_true,
            y_prob=y_prob,
            num_bins=int(config.get("holdout_num_bins", 100)),
            minimum_samples=int(config.get("holdout_minimum_samples", 10)),
            client_id=client_id,
            splits=splits,
        )
        print(
            f"[Client/Holdout] Learning set for the threshold: {summary['status']} | "
            f"client={client_id} | rows={summary['dataset']['rows']} | splits={splits}",
            flush=True,
        )
        return {
            "threshold_validation_status": summary["status"],
            "threshold_validation": json.dumps(summary, allow_nan=False),
        }
    except Exception as exc:
        print(f"[Client/Holdout] Learning set for the threshold failed: {exc}", flush=True)
        return {
            "threshold_validation_status": HOLDOUT_FAILED,
            "threshold_validation_error": str(exc),
        }


def run_local_holdout(
    config: Dict[str, Any],
    holdout_path: Optional[str],
    features: List[str],
    target: str,
    predict: Callable[[np.ndarray], np.ndarray],
    client_id: str,
    drop_missing_features: bool,
    validation: Optional[Callable[[], Tuple[np.ndarray, np.ndarray, Dict[str, int]]]] = None,
) -> Dict[str, Any]:
    """
    Run the holdout test on a client and return the entries to add to the
    Flower evaluate metrics: holdout_status, then holdout_result (JSON) or
    holdout_error.

    predict maps the holdout features to positive-class probabilities with
    the final global model.

    When the server optimizes the threshold, the learning-set summary is
    added too (see run_local_validation), even without a holdout file:
    the client learning set still helps choose the threshold.
    """
    entries = {}
    if bool(config.get("threshold_optimization_enabled", False)):
        entries = run_local_validation(config, validation, client_id)
    return {
        **entries,
        **_holdout_entries(config, holdout_path, features, target, predict, client_id, drop_missing_features),
    }


def _holdout_entries(
    config: Dict[str, Any],
    holdout_path: Optional[str],
    features: List[str],
    target: str,
    predict: Callable[[np.ndarray], np.ndarray],
    client_id: str,
    drop_missing_features: bool,
) -> Dict[str, Any]:
    if not holdout_path or not os.path.exists(holdout_path):
        print(f"[Client/Holdout] No holdout file on this client ({holdout_path})", flush=True)
        return {
            "holdout_status": HOLDOUT_MISSING,
            "holdout_client_id": client_id,
            "holdout_error": "No holdout file on this client",
        }

    try:
        X, y, dataset = load_holdout(holdout_path, features, target, drop_missing_features)
        probabilities = predict(X) if len(y) else np.empty((0,))
        summary = local_holdout_summary(
            y_true=y,
            y_prob=probabilities,
            threshold=float(config.get("holdout_threshold", 0.5)),
            num_bins=int(config.get("holdout_num_bins", 100)),
            minimum_samples=int(config.get("holdout_minimum_samples", 10)),
            client_id=client_id,
            dataset=dataset,
        )
        print(
            f"[Client/Holdout] {summary['status']} | client={client_id} | "
            f"rows={summary['dataset']['rows']} | metrics={summary.get('metrics')}",
            flush=True,
        )
        return {
            "holdout_status": summary["status"],
            "holdout_client_id": client_id,
            "holdout_result": json.dumps(summary, allow_nan=False),
        }
    except Exception as exc:
        print(f"[Client/Holdout] Holdout test failed: {exc}", flush=True)
        return {
            "holdout_status": HOLDOUT_FAILED,
            "holdout_client_id": client_id,
            "holdout_error": str(exc),
        }


# ----------------------------------------------------------------------
# Server side
# ----------------------------------------------------------------------


def pooled_roc_auc(pos_hist: np.ndarray, neg_hist: np.ndarray) -> Tuple[Optional[float], Optional[Dict[str, List[float]]]]:
    """
    ROC curve and AUC from score histograms.

    The thresholds are the bin edges. The AUC is the Mann-Whitney statistic
    where two scores in the same bin count as a tie (half a pair).
    """
    positives, negatives = pos_hist.sum(), neg_hist.sum()
    if positives == 0 or negatives == 0:
        return None, None

    # positives in strictly higher bins than each bin
    positives_above = np.concatenate([np.cumsum(pos_hist[::-1])[::-1][1:], [0]])
    auc = float((neg_hist * (positives_above + 0.5 * pos_hist)).sum() / (positives * negatives))

    # highest threshold first, starting at (0, 0)
    tpr = np.concatenate([[0.0], np.cumsum(pos_hist[::-1]) / positives])
    fpr = np.concatenate([[0.0], np.cumsum(neg_hist[::-1]) / negatives])
    return auc, {"fpr": [float(v) for v in fpr], "tpr": [float(v) for v in tpr]}


def _client_name(client_proxy) -> str:
    return str(getattr(client_proxy, "cid", client_proxy))


def histogram_confusion(pos_hist: np.ndarray, neg_hist: np.ndarray, edge: int) -> Dict[str, int]:
    """
    Exact confusion matrix at the threshold bins_edges[edge]: the scores at
    or above it are the bins edge, edge + 1, ... (np.histogram bins are
    [left, right) and the last one is closed).
    """
    pos_hist, neg_hist = np.asarray(pos_hist), np.asarray(neg_hist)
    return {
        "TP": int(pos_hist[edge:].sum()),
        "FP": int(neg_hist[edge:].sum()),
        "FN": int(pos_hist[:edge].sum()),
        "TN": int(neg_hist[:edge].sum()),
    }


def select_threshold_from_histograms(
    pos_hist: np.ndarray,
    neg_hist: np.ndarray,
    criterion: str,
    target_sensitivity: Optional[float] = None,
) -> Tuple[int, float]:
    """
    Choose the threshold among the inner bin edges of the score histograms
    and return (edge index, threshold).

    youden: maximizes sensitivity + specificity - 1, the middle edge of the
    best ones when several tie.
    target_sensitivity: the highest edge whose sensitivity reaches the
    target, so the best specificity under that constraint.
    """
    pos_hist, neg_hist = np.asarray(pos_hist), np.asarray(neg_hist)
    num_bins = len(pos_hist)
    positives, negatives = int(pos_hist.sum()), int(neg_hist.sum())
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"The learning-set data has a single class ({positives} positive, "
            f"{negatives} negative): the threshold cannot be optimized"
        )

    # candidate thresholds, strictly between 0 and 1
    edges = np.arange(1, num_bins)
    # scores in bin k and above
    positives_above = np.cumsum(pos_hist[::-1])[::-1]
    negatives_above = np.cumsum(neg_hist[::-1])[::-1]
    sensitivity = positives_above[edges] / positives
    specificity = 1 - negatives_above[edges] / negatives

    if criterion == "youden":
        youden = sensitivity + specificity - 1
        ties = np.flatnonzero(youden >= youden.max() - 1e-12)
        edge = int(edges[ties[len(ties) // 2]])
    elif criterion == "target_sensitivity":
        target = float(target_sensitivity)
        if not 0.0 < target <= 1.0:
            raise ValueError(f"The target sensitivity must be in (0, 1] (got {target})")
        # sensitivity decreases with the edge
        reached = np.flatnonzero(sensitivity >= target - 1e-12)
        if not len(reached):
            raise ValueError(
                f"No threshold reaches a learning-set sensitivity of {target}: "
                f"{int(pos_hist[0])} of the {positives} positive case(s) score below "
                f"the lowest candidate threshold {1 / num_bins:g}. Increase the number "
                f"of bins or lower the target sensitivity"
            )
        edge = int(edges[reached.max()])
    else:
        raise ValueError(f"Unknown threshold criterion '{criterion}'")

    return edge, float(_bin_edges(num_bins)[edge])


def optimize_federated_threshold(results, holdout_config: HoldoutConfig) -> Dict[str, Any]:
    """
    Choose the threshold on the pooled learning-set (train + validation +
    test) histograms of the final round. Returns the selection with status
    "completed" (threshold, edge, learning-set metrics) or "failed" (error),
    and the excluded clients.
    """
    completed, skipped, missing, failed = [], [], [], []

    for client_proxy, eval_res in results:
        metrics = eval_res.metrics or {}
        status = metrics.get("threshold_validation_status")
        name = str(metrics.get("holdout_client_id") or _client_name(client_proxy))

        if status is None:
            missing.append({
                "client_id": name,
                "error": "No learning-set data sent (the client MEDfl version may not support the optimized threshold)",
            })
            continue

        summary = None
        if metrics.get("threshold_validation"):
            try:
                summary = json.loads(metrics["threshold_validation"])
            except Exception as exc:
                failed.append({"client_id": name, "error": f"Invalid learning-set JSON: {exc}"})
                continue

        if summary and summary.get("data") != LEARNING_SET:
            # an older client sends only its validation split: never mix it with learning sets
            failed.append({
                "client_id": summary.get("client_id", name),
                "error": "This client sent only its validation split: update MEDfl on it to use its whole learning set",
            })
            continue

        if status == HOLDOUT_COMPLETED and summary:
            if int(summary.get("num_bins", -1)) != int(holdout_config.num_bins):
                failed.append({
                    "client_id": summary.get("client_id", name),
                    "error": f"Used {summary.get('num_bins')} histogram bins instead of {holdout_config.num_bins}",
                })
                continue
            completed.append(summary)
        elif status == HOLDOUT_SKIPPED and summary:
            skipped.append({
                "client_id": summary.get("client_id", name),
                "rows": summary["dataset"]["rows"],
                "reason": summary.get("reason"),
            })
        elif status == HOLDOUT_MISSING:
            missing.append({
                "client_id": name,
                "error": (summary or {}).get("reason") or metrics.get("threshold_validation_error"),
            })
        else:
            failed.append({
                "client_id": name,
                "error": metrics.get("threshold_validation_error", "No learning-set result returned"),
            })

    criterion = holdout_config.threshold_criterion
    selection: Dict[str, Any] = {
        "criterion": criterion,
        "criterion_label": THRESHOLD_CRITERIA.get(criterion, criterion),
        "target_sensitivity": (
            float(holdout_config.target_sensitivity) if criterion == "target_sensitivity" else None
        ),
        "data": LEARNING_SET,
        "method": f"pooled {int(holdout_config.num_bins)}-bin learning-set (train + validation + test) score histograms",
        "num_bins": int(holdout_config.num_bins),
        # the threshold is a bin edge
        "resolution": 1.0 / int(holdout_config.num_bins),
        "skipped_clients": skipped,
        "missing_clients": missing,
        "failed_clients": failed,
    }

    if not completed:
        return {**selection, "status": "failed", "error": "No client returned learning-set data"}

    pos_hist = np.sum([summary["pos_hist"] for summary in completed], axis=0)
    neg_hist = np.sum([summary["neg_hist"] for summary in completed], axis=0)
    positives, negatives = int(pos_hist.sum()), int(neg_hist.sum())
    selection["learning_set"] = {
        "clients": len(completed),
        "rows": positives + negatives,
        "positives": positives,
        "negatives": negatives,
        "splits": {
            split: int(sum(summary["dataset"].get("splits", {}).get(split, 0) for summary in completed))
            for split in LEARNING_SET_SPLITS
        },
    }

    try:
        edge, threshold = select_threshold_from_histograms(
            pos_hist, neg_hist, criterion, holdout_config.target_sensitivity
        )
    except ValueError as exc:
        return {**selection, "status": "failed", "error": str(exc)}

    confusion = histogram_confusion(pos_hist, neg_hist, edge)
    # learning-set ROC curve: its points are the bin edges, so the chosen threshold is one of them
    auc, roc = pooled_roc_auc(pos_hist, neg_hist)
    return {
        **selection,
        "status": "completed",
        "threshold": threshold,
        "edge": edge,
        "metrics": {"auc": auc, **confusion_metrics(confusion)},
        "confusion_matrix": confusion,
        "roc_curve": roc,
    }


def aggregate_federated_holdout(
    results,
    failures,
    holdout_config: HoldoutConfig,
    backend: str,
) -> Dict[str, Any]:
    """
    Aggregate the clients' holdout summaries of the final round.

    Args:
        results: Flower evaluate results, a list of (client_proxy, EvaluateRes)
        failures: Flower evaluate failures
        holdout_config: the server HoldoutConfig
        backend: "pytorch" or "xgboost"

    The "pooled" entry has the same keys as the simulation holdout results
    (dataset, threshold, metrics, confusion_matrix, roc_curve).
    """
    completed, skipped, missing, failed = [], [], [], []

    for client_proxy, eval_res in results:
        metrics = eval_res.metrics or {}
        status = metrics.get("holdout_status")
        # the client hostname when the client sent it, else the Flower client id
        name = str(metrics.get("holdout_client_id") or _client_name(client_proxy))

        summary = None
        if metrics.get("holdout_result"):
            try:
                summary = json.loads(metrics["holdout_result"])
            except Exception as exc:
                failed.append({"client_id": name, "error": f"Invalid holdout JSON: {exc}"})
                continue

        if status == HOLDOUT_COMPLETED and summary:
            if int(summary.get("num_bins", -1)) != int(holdout_config.num_bins):
                failed.append({
                    "client_id": summary.get("client_id", name),
                    "error": f"Used {summary.get('num_bins')} histogram bins instead of {holdout_config.num_bins}",
                })
                continue
            completed.append(summary)
        elif status == HOLDOUT_SKIPPED and summary:
            skipped.append({
                "client_id": summary.get("client_id", name),
                "rows": summary["dataset"]["rows"],
                "reason": summary.get("reason"),
            })
        elif status == HOLDOUT_MISSING:
            missing.append({"client_id": name, "error": metrics.get("holdout_error")})
        else:
            failed.append({
                "client_id": name,
                "error": metrics.get("holdout_error", "No holdout result returned"),
            })

    for failure in failures or []:
        if isinstance(failure, BaseException):
            failed.append({"client_id": None, "error": str(failure)})
        else:
            failed.append({"client_id": _client_name(failure[0]), "error": "Evaluation failed"})

    threshold = float(holdout_config.threshold)
    result: Dict[str, Any] = {
        "backend": backend,
        "threshold": threshold,
        "num_bins": int(holdout_config.num_bins),
        "minimum_samples": int(holdout_config.minimum_samples),
        "participating_clients": len(completed),
        "skipped_clients": skipped,
        "missing_clients": missing,
        "failed_clients": failed,
    }

    if holdout_config.threshold_source == "optimized":
        # chosen on the learning-set histograms only, the holdout ones are not read here
        selection = optimize_federated_threshold(results, holdout_config)
        result["threshold_selection"] = selection
        if selection["status"] != "completed":
            # no fallback to the manual threshold
            return {
                **result,
                "status": "failed",
                "error": f"Threshold optimization failed: {selection['error']}",
            }

        threshold = selection["threshold"]
        result["threshold"] = threshold
        # the client metrics were computed at the manual threshold: recompute them exactly at the bin edge
        for summary in completed:
            summary["threshold"] = threshold
            summary["confusion_matrix"] = histogram_confusion(
                summary["pos_hist"], summary["neg_hist"], selection["edge"]
            )
            summary["metrics"] = {
                "auc": summary["metrics"].get("auc"),
                **confusion_metrics(summary["confusion_matrix"]),
            }

    if not completed:
        return {
            **result,
            "status": "failed",
            "error": "No client returned a holdout result",
        }

    confusion = {
        key: int(sum(summary["confusion_matrix"][key] for summary in completed))
        for key in ("TP", "FP", "FN", "TN")
    }
    pos_hist = np.sum([summary["pos_hist"] for summary in completed], axis=0)
    neg_hist = np.sum([summary["neg_hist"] for summary in completed], axis=0)
    auc, roc = pooled_roc_auc(pos_hist, neg_hist)

    rows = sum(summary["dataset"]["rows"] for summary in completed)
    positives = sum(summary["dataset"]["positives"] for summary in completed)

    weighted = [
        (summary["dataset"]["rows"], summary["metrics"]["auc"])
        for summary in completed
        if summary["metrics"].get("auc") is not None
    ]
    auc_weighted_mean = (
        sum(n * value for n, value in weighted) / sum(n for n, _ in weighted)
        if weighted
        else None
    )

    result.update({
        "status": "completed",
        "pooled": {
            "dataset": {
                "name": f"Federated holdout ({len(completed)} client(s))",
                "rows": int(rows),
                "positives": int(positives),
                "negatives": int(rows - positives),
                "rows_without_target": int(
                    sum(summary["dataset"]["rows_without_target"] for summary in completed)
                ),
            },
            "threshold": threshold,
            "metrics": {"auc": auc, **confusion_metrics(confusion)},
            "confusion_matrix": confusion,
            "roc_curve": roc,
        },
        "auc_method": f"pooled from {int(holdout_config.num_bins)}-bin score histograms",
        "auc_weighted_mean": auc_weighted_mean,
    })

    if holdout_config.include_client_results:
        result["clients"] = [
            {
                "client_id": summary["client_id"],
                "dataset": summary["dataset"],
                "metrics": summary["metrics"],
                "confusion_matrix": summary["confusion_matrix"],
            }
            for summary in completed
        ]

    return result


# The app reads the result from this log line (the XGBoost server answers
# its caller before training ends)
HOLDOUT_LOG_PREFIX = "[Server/Holdout] FEDERATED_HOLDOUT_RESULT "


def finalize_federated_holdout(
    results,
    failures,
    holdout_config: HoldoutConfig,
    backend: str,
) -> Dict[str, Any]:
    """
    Aggregate the final-round holdout results without ever stopping the
    server, then print them on one line prefixed by HOLDOUT_LOG_PREFIX.
    """
    try:
        result = aggregate_federated_holdout(results, failures, holdout_config, backend)
    except Exception as exc:
        result = {"status": "failed", "backend": backend, "error": f"Holdout aggregation failed: {exc}"}

    print(
        f"[Server/Holdout] Federated holdout {result['status']} | "
        f"clients={result.get('participating_clients', 0)} | "
        f"metrics={(result.get('pooled') or {}).get('metrics')}",
        flush=True,
    )
    print(HOLDOUT_LOG_PREFIX + json.dumps(result, allow_nan=False, default=str), flush=True)
    return result
