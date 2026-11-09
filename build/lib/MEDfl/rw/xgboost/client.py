import argparse
import json
import socket
import platform
import shutil
from typing import Dict, Any, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import psutil
import xgboost as xgb
import torch
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Status,
)

try:
    import GPUtil
except ImportError:
    GPUtil = None

from MEDfl.rw.identity import client_name
from MEDfl.rw.xgboost.data import XGBoostDataModule
from MEDfl.rw.xgboost.metrics import evaluate_predictions
from MEDfl.rw.xgboost.serialization import (
    booster_to_parameters,
    parameters_to_booster,
)
from MEDfl.LearningManager.shap import (
    LocalXGBoostSHAPExplainer,
)
from MEDfl.rw.holdout import (
    HOLDOUT_UNSUPPORTED,
    run_local_holdout,
)


class XGBoostFlowerClient(fl.client.Client):
    """
    Real-world Flower client for federated XGBoost in MEDfl.

    This client is intentionally separate from the existing PyTorch NumPyClient.
    The PyTorch client exchanges neural network weights. This client exchanges
    serialized XGBoost boosters.

    Parameters
    ----------
    server_address : str
        Flower server address, e.g. "127.0.0.1:8080".
    data_path : str
        Local CSV dataset path.
    val_frac : Optional[float]
        Optional client-side validation fraction override.
    test_frac : Optional[float]
        Optional client-side test fraction override.
    id_col : Optional[str]
        Column used for ID-based test split.
    test_ids : Optional[str]
        Comma-separated test IDs.
    seed : int
        Random seed.
    holdout_path : Optional[str]
        Optional local holdout CSV, never used for training. When the server
        enables the federated holdout test, the final global booster is
        tested on it after the last round.
    """

    def __init__(
        self,
        server_address: str,
        data_path: str = "data/data.csv",
        val_frac: Optional[float] = None,
        test_frac: Optional[float] = None,
        id_col: Optional[str] = None,
        test_ids: Optional[str] = None,
        seed: int = 42,
        holdout_path: Optional[str] = None,
    ):
        self.server_address = server_address
        self.data_path = data_path
        # read only when the server asks for the holdout test (features/target come from the server)
        self.holdout_path = holdout_path

        self.data_module = XGBoostDataModule(
            data_path=data_path,
            val_frac=val_frac,
            test_frac=test_frac,
            id_col=id_col,
            test_ids=test_ids,
            seed=seed,
        )

        self.booster: Optional[xgb.Booster] = None

        self.dtrain = None
        self.dval = None
        self.dtest = None
        self.metadata: Dict[str, Any] = {}

        self.task = "binary"
        self.threshold = 0.5
        self.xgb_params: Dict[str, Any] = {}
        self.local_num_boost_round = 10

        self._initialized = False

        self._df_preview = pd.read_csv(data_path, nrows=5)

    def _initialize_from_config(self, config: Dict[str, Any]) -> None:
        """
        Lazily initialize DMatrix objects from server config.
        """

        self.task = str(config.get("task", self.task) or self.task)
        self.threshold = float(config.get("threshold", self.threshold))
        self.local_num_boost_round = int(
            config.get("local_num_boost_round", self.local_num_boost_round)
        )

        xgb_params_raw = config.get("xgb_params", "{}")

        if isinstance(xgb_params_raw, str):
            try:
                self.xgb_params = json.loads(xgb_params_raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid xgb_params JSON received from server: {xgb_params_raw}"
                ) from exc
        elif isinstance(xgb_params_raw, dict):
            self.xgb_params = xgb_params_raw
        else:
            raise ValueError(
                "xgb_params must be either a JSON string or a Python dictionary."
            )

        self.dtrain, self.dval, self.dtest, self.metadata = (
            self.data_module.prepare_from_config(config)
        )

        self._initialized = True

        print(
            "[XGBoostClient] Initialized with "
            f"task={self.task}, "
            f"features={self.metadata.get('features')}, "
            f"target={self.metadata.get('target')}, "
            f"train={self.metadata.get('num_train')}, "
            f"val={self.metadata.get('num_val')}, "
            f"test={self.metadata.get('num_test')}"
        )

    # ------------------------------------------------------------------
    # Federated SHAP helpers
    # ------------------------------------------------------------------

    def _get_shap_dmatrix(self, data_split: str) -> xgb.DMatrix:
        """
        Return the local DMatrix selected for the post-training SHAP phase.
        """
        split = str(data_split).lower()

        if split == "train":
            matrix = self.dtrain
        elif split == "validation":
            matrix = self.dval
        elif split == "test":
            matrix = self.dtest
        else:
            raise ValueError(
                f"Unsupported SHAP data split '{data_split}'. "
                "Expected train, validation, or test."
            )

        if matrix is None:
            raise ValueError(
                f"The selected SHAP split '{split}' is not available "
                "on this XGBoost client."
            )

        if int(matrix.num_row()) <= 0:
            raise ValueError(
                f"The selected SHAP split '{split}' is empty "
                "on this XGBoost client."
            )

        return matrix

    @staticmethod
    def _dmatrix_to_shap_loader(
        matrix: xgb.DMatrix,
        batch_size: int = 256,
    ) -> DataLoader:
        """
        Convert a local XGBoost DMatrix into the DataLoader interface expected
        by LocalXGBoostSHAPExplainer.

        The conversion is local only. No samples are transmitted.
        """
        if matrix is None:
            raise ValueError("Cannot create a SHAP loader from None.")

        try:
            features = matrix.get_data()
        except AttributeError as exc:
            raise RuntimeError(
                "This XGBoost version does not expose DMatrix.get_data(). "
                "Upgrade XGBoost or adapt XGBoostDataModule to retain the "
                "local NumPy feature splits for SHAP."
            ) from exc

        # DMatrix.get_data() commonly returns a scipy sparse matrix.
        if hasattr(features, "toarray"):
            features = features.toarray()

        features = np.asarray(features, dtype=np.float32)

        if features.ndim != 2:
            raise ValueError(
                "SHAP expects 2-D tabular XGBoost features. "
                f"Received shape {features.shape}."
            )

        labels = np.asarray(
            matrix.get_label(),
            dtype=np.float32,
        )

        x_tensor = torch.tensor(
            features,
            dtype=torch.float32,
        )
        y_tensor = torch.tensor(
            labels,
            dtype=torch.float32,
        )

        return DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=max(1, min(int(batch_size), len(x_tensor))),
            shuffle=False,
        )

    def _calculate_local_shap(
        self,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Explain the final federated XGBoost booster on this client's selected
        local data partition.

        Only additive SHAP summaries are serialized back to the server.
        """
        if not self._initialized:
            self._initialize_from_config(config)

        if self.booster is None:
            raise RuntimeError(
                "Cannot calculate XGBoost SHAP without a booster."
            )

        explainer_name = str(
            config.get("shap_explainer", "tree")
        ).lower()

        if explainer_name != "tree":
            raise ValueError(
                "Real-world XGBoost SHAP requires explainer='tree'."
            )

        data_split = str(
            config.get("shap_data_split", "validation")
        )
        background_size = int(
            config.get("shap_background_size", 100)
        )
        explanation_size = int(
            config.get("shap_explanation_size", 500)
        )
        random_seed = int(
            config.get("shap_random_seed", 42)
        )
        minimum_samples = int(
            config.get("shap_minimum_samples", 10)
        )
        model_output = str(
            config.get("shap_model_output", "raw")
        )

        clipping_raw = float(
            config.get("shap_clipping_value", -1.0)
        )
        clipping_value = (
            None if clipping_raw <= 0 else clipping_raw
        )

        feature_names_raw = str(
            config.get("shap_feature_names_json", "[]")
        )

        try:
            configured_feature_names = json.loads(
                feature_names_raw
            )
        except Exception:
            configured_feature_names = []

        effective_features = list(
            self.metadata.get("features", [])
        )

        feature_names = (
            list(configured_feature_names)
            if configured_feature_names
            else effective_features
        )

        if len(feature_names) != int(
            self.metadata.get("num_features", 0)
        ):
            raise ValueError(
                "SHAP feature names do not match the XGBoost input size. "
                f"Expected {self.metadata.get('num_features', 0)}, "
                f"received {len(feature_names)}."
            )

        matrix = self._get_shap_dmatrix(data_split)

        available_samples = int(matrix.num_row())

        if available_samples < minimum_samples:
            raise ValueError(
                f"Only {available_samples} samples are available in "
                f"the '{data_split}' split, but "
                f"minimum_samples={minimum_samples}."
            )

        loader = self._dmatrix_to_shap_loader(
            matrix=matrix,
            batch_size=min(256, available_samples),
        )

        local_explainer = LocalXGBoostSHAPExplainer(
            background_size=background_size,
            explanation_size=explanation_size,
            random_seed=random_seed,
            clipping_value=clipping_value,
            model_output=model_output,
        )

        result = local_explainer.calculate(
            booster=self.booster,
            loader=loader,
        )

        explained_samples = int(
            result["sample_count"]
        )

        if explained_samples < minimum_samples:
            raise ValueError(
                f"Only {explained_samples} samples were explained, "
                f"but minimum_samples={minimum_samples}."
            )

        return {
            "client_id": client_name(),
            "feature_names": feature_names,
            "available_samples": available_samples,
            "sample_count": explained_samples,
            "abs_sum": np.asarray(
                result["abs_sum"],
                dtype=np.float64,
            ).tolist(),
            "signed_sum": np.asarray(
                result["signed_sum"],
                dtype=np.float64,
            ).tolist(),
            "square_sum": np.asarray(
                result["square_sum"],
                dtype=np.float64,
            ).tolist(),
            "positive_count": np.asarray(
                result["positive_count"],
                dtype=np.int64,
            ).tolist(),
            "negative_count": np.asarray(
                result["negative_count"],
                dtype=np.int64,
            ).tolist(),
            "zero_count": np.asarray(
                result["zero_count"],
                dtype=np.int64,
            ).tolist(),
        }

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """
        Return current booster parameters.

        Before training, this can be empty.
        """

        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=booster_to_parameters(self.booster),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """
        Receive global booster, continue local XGBoost training, return updated booster.
        """

        config = dict(ins.config or {})

        if not self._initialized:
            self._initialize_from_config(config)

        received_booster = parameters_to_booster(ins.parameters)

        if received_booster is not None:
            self.booster = received_booster

        evals = []
        if self.dval is not None and self.metadata.get("num_val", 0) > 0:
            evals.append((self.dval, "validation"))

        self.booster = xgb.train(
            params=self.xgb_params,
            dtrain=self.dtrain,
            num_boost_round=self.local_num_boost_round,
            xgb_model=self.booster,
            evals=evals,
            verbose_eval=False,
        )

        train_pred = self.booster.predict(self.dtrain)

        train_metrics = evaluate_predictions(
            task=self.task,
            y_true=self.dtrain.get_label(),
            y_pred_or_prob=train_pred,
            threshold=self.threshold,
            num_features=int(self.metadata.get("num_features", 0)),
        )

        metrics = {
            "backend": "xgboost",
            "task": self.task,
            "hostname": client_name(),
            "os_type": platform.system(),
            "num_train": int(self.metadata.get("num_train", 0)),
            "features": ",".join(self.metadata.get("features", [])),
            "target": str(self.metadata.get("target", "")),
            "val_fraction": float(self.metadata.get("val_fraction", 0.0)),
            "test_fraction": float(self.metadata.get("test_fraction", 0.0)),
        }

        for key, value in train_metrics.items():
            metrics[f"train_{key}"] = value

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=booster_to_parameters(self.booster),
            num_examples=int(self.metadata.get("num_train", 0)),
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evaluate received global booster on local test set.
        """

        config = dict(ins.config or {})

        if not self._initialized:
            self._initialize_from_config(config)

        received_booster = parameters_to_booster(ins.parameters)

        if received_booster is not None:
            self.booster = received_booster

        if self.booster is None:
            return EvaluateRes(
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="No booster."),
                loss=0.0,
                num_examples=0,
                metrics={},
            )

        y_true = self.dtest.get_label()
        y_pred = self.booster.predict(self.dtest)

        eval_metrics = evaluate_predictions(
            task=self.task,
            y_true=y_true,
            y_pred_or_prob=y_pred,
            threshold=self.threshold,
            num_features=int(self.metadata.get("num_features", 0)),
        )

        metrics = {
            "backend": "xgboost",
            "task": self.task,
        }

        for key, value in eval_metrics.items():
            metrics[f"eval_{key}"] = value

        if self.task == "regression":
            loss = float(eval_metrics.get("rmse", 0.0))
        elif self.task == "binary":
            loss = float(eval_metrics.get("logloss", 0.0))
        else:
            loss = float(eval_metrics.get("mlogloss", 0.0))

        # --------------------------------------------------------------
        # Final-round federated SHAP.
        #
        # ins.parameters contains the server-aggregated booster for this
        # evaluation. Therefore all participating clients explain the
        # same final global XGBoost model.
        # --------------------------------------------------------------
        if bool(config.get("shap_enabled", False)):
            print(
                "[XGBoostClient/SHAP] Starting local SHAP on "
                "the final global booster...",
                flush=True,
            )

            try:
                local_shap_result = (
                    self._calculate_local_shap(config)
                )

                metrics["shap_status"] = "completed"
                metrics["shap_result"] = json.dumps(
                    local_shap_result,
                    allow_nan=False,
                )

                print(
                    "[XGBoostClient/SHAP] Local SHAP completed | "
                    f"client={local_shap_result['client_id']} | "
                    f"samples={local_shap_result['sample_count']}",
                    flush=True,
                )

            except Exception as exc:
                metrics["shap_status"] = "failed"
                metrics["shap_error"] = str(exc)

                print(
                    "[XGBoostClient/SHAP] Local SHAP failed: "
                    f"{exc}",
                    flush=True,
                )

        # --------------------------------------------------------------
        # Final-round federated holdout test on the same final booster.
        # Missing feature values are kept: XGBoost handles them.
        # --------------------------------------------------------------
        if bool(config.get("holdout_enabled", False)):
            if self.task != "binary":
                metrics["holdout_status"] = HOLDOUT_UNSUPPORTED
                metrics["holdout_client_id"] = client_name()
                metrics["holdout_error"] = (
                    "The holdout test supports binary classification "
                    f"only (task is '{self.task}')"
                )
            else:
                metrics.update(
                    run_local_holdout(
                        config=config,
                        holdout_path=self.holdout_path,
                        features=self.data_module.effective_features,
                        target=self.data_module.effective_target,
                        predict=self._predict_probabilities,
                        client_id=client_name(),
                        drop_missing_features=False,
                        validation=self._validation_predictions,
                    )
                )

        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=loss,
            num_examples=int(self.metadata.get("num_test", 0)),
            metrics=metrics,
        )

    def _predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Positive-class probabilities of the current booster for a feature array."""
        matrix = xgb.DMatrix(
            X,
            feature_names=list(self.data_module.effective_features),
        )
        return np.asarray(self.booster.predict(matrix), dtype=float)

    def _validation_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """(y_true, y_prob) of the validation split, for the optimized holdout threshold."""
        if self.dval is None or int(self.metadata.get("num_val", 0)) == 0:
            return np.empty((0,)), np.empty((0,))
        return (
            np.asarray(self.dval.get_label(), dtype=float),
            np.asarray(self.booster.predict(self.dval), dtype=float),
        )

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        """
        Return machine and dataset properties.
        """

        hostname = client_name()
        os_type = platform.system()

        df = self.data_module.df

        feature_names = df.columns[:-1].tolist()
        target_name = df.columns[-1]

        try:
            label_counts = df[target_name].value_counts().to_dict()
            dist_str = ",".join(f"{k}:{v}" for k, v in label_counts.items())
            classes_str = ",".join(map(str, sorted(label_counts.keys())))
        except Exception:
            dist_str = ""
            classes_str = ""

        cpu_physical = psutil.cpu_count(logical=False)
        cpu_logical = psutil.cpu_count(logical=True)
        total_mem_gb = round(psutil.virtual_memory().total / (1024**3), 2)

        driver_present = shutil.which("nvidia-smi") is not None
        gpu_count = 0

        if GPUtil and driver_present:
            try:
                gpu_count = len(GPUtil.getGPUs())
            except Exception:
                gpu_count = 0

        properties = {
            "backend": "xgboost",
            "hostname": hostname,
            "os_type": os_type,
            "num_samples": int(len(df)),
            "num_features": int(len(feature_names)),
            "features": ",".join(feature_names),
            "target": target_name,
            "classes": classes_str,
            "label_distribution": dist_str,
            "cpu_physical_cores": int(cpu_physical or 0),
            "cpu_logical_cores": int(cpu_logical or 0),
            "total_memory_gb": float(total_mem_gb),
            "gpu_driver_present": str(driver_present),
            "gpu_count": int(gpu_count),
        }

        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties=properties,
        )

    def start(self) -> None:
        """
        Start the Flower client.
        """

        fl.client.start_client(
            server_address=self.server_address,
            client=self,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MEDfl real-world XGBoost client")

    parser.add_argument(
        "--server_address",
        type=str,
        required=True,
        help="Flower server address, e.g. 127.0.0.1:8080",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/data.csv",
        help="Path to local CSV dataset.",
    )
    parser.add_argument("--val_frac", type=float, default=None)
    parser.add_argument("--test_frac", type=float, default=None)
    parser.add_argument("--id_col", type=str, default=None)
    parser.add_argument("--test_ids", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--holdout_path",
        type=str,
        default=None,
        help="Optional local holdout CSV for the federated holdout test.",
    )

    args = parser.parse_args()

    client = XGBoostFlowerClient(
        server_address=args.server_address,
        data_path=args.data_path,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        id_col=args.id_col,
        test_ids=args.test_ids,
        seed=args.seed,
        holdout_path=args.holdout_path,
    )

    client.start()


if __name__ == "__main__":
    main()