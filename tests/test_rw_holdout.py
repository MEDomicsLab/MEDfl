# test_rw_holdout.py
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from MEDfl.rw.holdout import (
    HoldoutConfig,
    aggregate_federated_holdout,
    confusion_metrics,
    load_holdout,
    local_holdout_summary,
    pooled_roc_auc,
    run_local_holdout,
)


def evaluate_result(cid, metrics):
    """(client_proxy, EvaluateRes)-like pair as Flower passes to aggregate_evaluate."""
    return SimpleNamespace(cid=cid), SimpleNamespace(metrics=metrics)


def client_metrics(summary):
    return {"holdout_status": summary["status"], "holdout_result": json.dumps(summary, allow_nan=False)}


class TestFederatedHoldout(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.config = HoldoutConfig(enabled=True, threshold=0.4, num_bins=100, minimum_samples=10)
        # 3 clients with different sizes and prevalences
        self.clients = []
        for size, prevalence in [(300, 0.2), (150, 0.5), (80, 0.35)]:
            y = (rng.random(size) < prevalence).astype(float)
            prob = np.clip(0.3 * y + rng.normal(0.35, 0.2, size), 0, 1)
            self.clients.append((y, prob))

    def summaries(self):
        return [
            local_holdout_summary(y, prob, self.config.threshold, self.config.num_bins,
                                  self.config.minimum_samples, f"client{i}", {"name": "holdout.csv", "rows_without_target": 0})
            for i, (y, prob) in enumerate(self.clients)
        ]

    def test_pooled_matches_centralized(self):
        results = [evaluate_result(str(i), client_metrics(s)) for i, s in enumerate(self.summaries())]
        result = aggregate_federated_holdout(results, [], self.config, "pytorch")

        y = np.concatenate([y for y, _ in self.clients])
        prob = np.concatenate([p for _, p in self.clients])
        pred = prob >= self.config.threshold
        central = {
            "TP": int((pred & (y == 1)).sum()), "FP": int((pred & (y == 0)).sum()),
            "FN": int((~pred & (y == 1)).sum()), "TN": int((~pred & (y == 0)).sum()),
        }

        pooled = result["pooled"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["participating_clients"], 3)
        self.assertEqual(pooled["confusion_matrix"], central)
        for key, value in confusion_metrics(central).items():
            self.assertAlmostEqual(pooled["metrics"][key], value, places=12)
        self.assertAlmostEqual(pooled["metrics"]["auc"], roc_auc_score(y, prob), delta=0.01)
        self.assertEqual(pooled["dataset"]["rows"], len(y))
        self.assertEqual(pooled["dataset"]["positives"], int(y.sum()))
        self.assertEqual(len(result["clients"]), 3)
        self.assertNotIn("pos_hist", result["clients"][0])
        # the result is sent to the app as JSON
        json.dumps(result, allow_nan=False)

    def test_histogram_auc(self):
        y, prob = self.clients[0]
        edges = np.linspace(0, 1, 101)
        auc, roc = pooled_roc_auc(np.histogram(prob[y == 1], edges)[0], np.histogram(prob[y == 0], edges)[0])
        self.assertAlmostEqual(auc, roc_auc_score(y, prob), delta=0.01)
        self.assertEqual((roc["fpr"][0], roc["tpr"][0]), (0.0, 0.0))
        self.assertAlmostEqual(roc["fpr"][-1], 1.0)
        self.assertAlmostEqual(roc["tpr"][-1], 1.0)
        # a single class has no ROC curve
        self.assertEqual(pooled_roc_auc(np.array([0, 3]), np.array([0, 0])), (None, None))

    def test_client_statuses(self):
        summaries = self.summaries()
        small = local_holdout_summary(np.array([0.0, 1.0]), np.array([0.2, 0.8]), 0.4, 100, 10, "small", {"name": "h.csv", "rows_without_target": 0})
        other_bins = local_holdout_summary(*self.clients[1], 0.4, 50, 10, "bins", {"name": "h.csv", "rows_without_target": 0})
        results = [
            evaluate_result("0", client_metrics(summaries[0])),
            evaluate_result("1", client_metrics(small)),
            evaluate_result("2", {"holdout_status": "missing", "holdout_client_id": "host2", "holdout_error": "No holdout file on this client"}),
            evaluate_result("3", {"holdout_status": "failed", "holdout_error": "boom"}),
            evaluate_result("4", {"holdout_status": "completed", "holdout_result": "{not json"}),
            evaluate_result("5", client_metrics(other_bins)),
            evaluate_result("6", {"eval_auc": 0.7}),
        ]
        result = aggregate_federated_holdout(results, [RuntimeError("lost")], self.config, "xgboost")

        self.assertEqual(small["status"], "skipped")
        self.assertNotIn("pos_hist", small)
        self.assertEqual(result["participating_clients"], 1)
        self.assertEqual([c["client_id"] for c in result["skipped_clients"]], ["small"])
        self.assertEqual([c["client_id"] for c in result["missing_clients"]], ["host2"])
        self.assertEqual([c["client_id"] for c in result["failed_clients"]], ["3", "4", "bins", "6", None])
        self.assertEqual(result["pooled"]["confusion_matrix"], summaries[0]["confusion_matrix"])

    def test_no_result(self):
        results = [evaluate_result("0", {"holdout_status": "missing"})]
        result = aggregate_federated_holdout(results, [], self.config, "pytorch")
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("pooled", result)


class TestLoadHoldout(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.path = os.path.join(self.folder, "holdout.csv")
        pd.DataFrame({
            "b": [1.0, 2.0, np.nan, 4.0, 5.0],
            "a": [0.1, 0.2, 0.3, 0.4, 0.5],
            "label": [0, 1, 1, None, 0],
        }).to_csv(self.path, index=False)

    def tearDown(self):
        for name in os.listdir(self.folder):
            os.remove(os.path.join(self.folder, name))
        os.rmdir(self.folder)

    def test_feature_order_and_rows(self):
        X, y, info = load_holdout(self.path, ["a", "b"], "label", drop_missing_features=True)
        np.testing.assert_allclose(X, [[0.1, 1.0], [0.2, 2.0], [0.5, 5.0]], rtol=1e-6)
        np.testing.assert_array_equal(y, [0, 1, 0])
        self.assertEqual(info["rows_without_target"], 1)
        self.assertEqual(info["rows_with_missing_features"], 1)

        X, y, _ = load_holdout(self.path, ["a", "b"], "label", drop_missing_features=False)
        self.assertEqual(len(y), 4)
        self.assertTrue(np.isnan(X[2, 1]))

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "missing 1 training column"):
            load_holdout(self.path, ["a", "c"], "label", True)
        with self.assertRaisesRegex(ValueError, "no 'target' column"):
            load_holdout(self.path, ["a"], "target", True)
        pd.DataFrame({"a": [1, 2], "label": [1, 2]}).to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "must be binary"):
            load_holdout(self.path, ["a"], "label", True)

    def test_run_local_holdout(self):
        config = {"holdout_threshold": 0.5, "holdout_num_bins": 10, "holdout_minimum_samples": 1}
        missing = run_local_holdout(config, os.path.join(self.folder, "none.csv"), ["a"], "label", None, "c", True)
        self.assertEqual(missing["holdout_status"], "missing")

        completed = run_local_holdout(config, self.path, ["a", "b"], "label", lambda X: X[:, 0] * 2, "c", True)
        self.assertEqual(completed["holdout_status"], "completed")
        summary = json.loads(completed["holdout_result"])
        self.assertEqual(summary["dataset"]["rows"], 3)
        self.assertEqual(summary["metrics"]["f1"], 0.0)
        self.assertEqual(sum(summary["pos_hist"]) + sum(summary["neg_hist"]), 3)

        def broken(X):
            raise RuntimeError("model error")
        failed = run_local_holdout(config, self.path, ["a", "b"], "label", broken, "c", True)
        self.assertEqual((failed["holdout_status"], failed["holdout_error"]), ("failed", "model error"))


if __name__ == "__main__":
    unittest.main()
