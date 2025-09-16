# client.py

import argparse
import pandas as pd
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from MEDfl.rw.model import Net  # votre définition de modèle
import socket
import platform
import psutil
import shutil
import numpy as np

try:
    import GPUtil
except ImportError:
    GPUtil = None

try:
    import xgboost as xgb
except Exception:
    xgb = None


class DPConfig:
    """
    Configuration for differential privacy.

    Attributes:
        noise_multiplier (float): Noise multiplier for DP.
        max_grad_norm (float): Maximum gradient norm for clipping.
        batch_size (int): Batch size for training.
        secure_rng (bool): Use a secure random generator.
    """

    def __init__(
        self,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        batch_size: int = 32,
        secure_rng: bool = False,
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.secure_rng = secure_rng


def booster_to_parameters(bst):
    # Send JSON so the server can merge trees safely
    raw = bst.save_raw(raw_format="json")
    # NumPyClient expects a list of numpy arrays
    return [np.frombuffer(raw, dtype=np.uint8)]


def parameters_to_booster(parameters):
    # Handle numpy uint8 array or raw bytes/bytearray
    buf = parameters[0]
    if isinstance(buf, (bytes, bytearray, memoryview)):
        raw = bytes(buf)
    else:
        arr = np.asarray(buf, dtype=np.uint8)
        raw = arr.tobytes()

    booster = xgb.Booster()
    # xgboost can load both JSON and binary buffers
    booster.load_model(bytearray(raw))
    return booster


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        server_address: str,
        data_path: str = "data/data.csv",
        dp_config: DPConfig = None,
        model_type: str = "nn",
        xgb_params: dict = None,
        xgb_rounds: int = 50,
    ):
        self.server_address = server_address
        self.model_type = model_type.lower()
        self.xgb_params = xgb_params or {}
        self.xgb_rounds = xgb_rounds

        # Store hostname for datasetConfig host-specific overrides
        self.hostname = socket.gethostname()

        # Load once; keep the DataFrame so we can rebuild splits from config later
        self.df = pd.read_csv(data_path)

        # Defaults at startup (can be overridden dynamically from config)
        default_target = self.df.columns[-1]
        default_test_size = 0.20

        # Build initial splits/buffers/metadata
        self._prepare_splits(
            target_name=default_target,
            test_size=default_test_size,
            dp_config=dp_config,
        )

        # Apply DP once at startup (NN only)
        self.privacy_engine = None
        if dp_config and self.model_type == "nn":
            try:
                from opacus import PrivacyEngine

                self.privacy_engine = PrivacyEngine()
                (
                    self.model,
                    self.optimizer,
                    self.train_loader,
                ) = self.privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=dp_config.noise_multiplier,
                    max_grad_norm=dp_config.max_grad_norm,
                    secure_rng=dp_config.secure_rng,
                )
            except ImportError:
                print("Opacus non installé : exécution sans DP.")

    # --------------------------------------------------------------------------
    # Helpers to (re)prepare data from a target column and test_size
    # --------------------------------------------------------------------------
    def _prepare_splits(self, target_name, test_size, dp_config):
        """Create train/test split, tensors/DMatrices, and metadata for the chosen target/test_size."""
        df = self.df

        if target_name not in df.columns:
            raise ValueError(f"Target '{target_name}' not found in CSV columns: {list(df.columns)}")

        # Clamp and sanitize test_size
        try:
            ts = float(test_size)
        except Exception:
            ts = 0.20
        ts = max(1e-6, min(ts, 0.9))

        # Build X/y from chosen target
        X_df = df.drop(columns=[target_name])
        y_series = df[target_name]
        X_full = X_df.values

        # If y isn't numeric, factorize to integers (keep class labels for metadata)
        if not np.issubdtype(np.asarray(y_series).dtype, np.number):
            y_vals, uniques = pd.factorize(y_series)
            y_full = y_vals.astype(np.float32, copy=False)
            classes = list(map(str, uniques))
            label_counts = y_series.value_counts().to_dict()
        else:
            y_full = y_series.values.astype(np.float32, copy=False)
            classes = sorted(pd.Series(y_full).unique().tolist())
            label_counts = pd.Series(y_full).value_counts().to_dict()

        # Heuristic for stratification (classification-like)
        is_classif = np.unique(y_full).shape[0] <= 50
        strat = y_full if is_classif else None

        X_train, X_test, y_train, y_test = train_test_split(
            X_full,
            y_full,
            test_size=ts,
            random_state=42,
            stratify=strat if strat is not None else None,
        )

        # --- Update metadata used by get_properties ---
        self.feature_names = X_df.columns.tolist()
        self.target_name = target_name
        self.label_counts = label_counts
        self.classes = classes

        # --- Build per-model buffers ---
        if self.model_type == "nn":
            # Train tensors
            self.X_tensor = torch.tensor(X_train, dtype=torch.float32)
            self.y_tensor = torch.tensor(y_train, dtype=torch.float32)

            # Test tensors
            self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            self.y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

            # DataLoaders
            batch_size = getattr(dp_config, "batch_size", 32) if dp_config else 32
            train_ds = TensorDataset(self.X_tensor, self.y_tensor)
            self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(
                TensorDataset(self.X_test_tensor, self.y_test_tensor),
                batch_size=batch_size,
                shuffle=False,
            )

            # Create model/criterion/optimizer if not present
            input_dim = X_train.shape[1]
            if not hasattr(self, "model"):
                self.model = Net(input_dim)
                self.criterion = nn.BCEWithLogitsLoss()
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        else:
            # XGBoost DMatrices
            if xgb is None:
                raise ImportError("xgboost is not installed. `pip install xgboost`")

            self.X_np = X_train.astype(np.float32, copy=False)   # train
            self.y_np = y_train.astype(np.float32, copy=False)
            self.dtrain = xgb.DMatrix(self.X_np, label=self.y_np)

            self.X_np_test = X_test.astype(np.float32, copy=False)  # test
            self.y_np_test = y_test.astype(np.float32, copy=False)
            self.dtest = xgb.DMatrix(self.X_np_test, label=self.y_np_test)

            # Cold-start booster only if not present yet
            if not hasattr(self, "bst"):
                self.bst = xgb.train(self.xgb_params, self.dtrain, num_boost_round=1)

        # Remember current prep so we can skip unnecessary rebuilds
        self._prepared_key = (self.target_name, float(ts))

    def _pick_target_and_frac(self, cfg):
        """Pick ('target', 'testFrac') from a small dict; support alternative keys."""
        if not isinstance(cfg, dict):
            return None, None
        tgt = cfg.get("target") or cfg.get("Target") or cfg.get("label")
        frac = cfg.get("testFrac", cfg.get("test_size", None))
        try:
            frac = float(frac) if frac is not None else None
        except Exception:
            frac = None
        return tgt, frac

    def _resolve_dataset_from_cfg(self, ds_cfg):
        """
        Resolve (target, test_size) from datasetConfig:
          - If isGlobal=True, take globalConfig.{target,testFrac}
          - else take datasetConfig[hostname].{target,testFrac}
          - fallbacks: keep current settings or defaults
        """
        default_target = getattr(self, "target_name", self.df.columns[-1])
        default_frac = getattr(self, "_prepared_key", (None, 0.2))[1] if hasattr(self, "_prepared_key") else 0.2

        if not isinstance(ds_cfg, dict):
            return default_target, default_frac

        is_global = bool(ds_cfg.get("isGlobal"))
        if is_global:
            tgt, frac = self._pick_target_and_frac(ds_cfg.get("globalConfig", {}))
        else:
            host_cfg = ds_cfg.get(self.hostname)
            if not isinstance(host_cfg, dict):
                lower_map = {str(k).lower(): v for k, v in ds_cfg.items() if isinstance(v, dict)}
                host_cfg = lower_map.get(self.hostname.lower())
            tgt, frac = self._pick_target_and_frac(host_cfg or {})

        target = tgt if tgt else default_target
        test_size = frac if frac is not None else default_frac
        test_size = max(1e-6, min(float(test_size), 0.9))
        return target, test_size

    def _ensure_prepared_from_config(self, config, dp_config=None):
        """Check config['datasetConfig']; rebuild splits if (target,test_size) changed."""
        ds_cfg = config.get("dataset_config") if isinstance(config, dict) else None
        if not isinstance(ds_cfg, dict):
            return

        target, ts = self._resolve_dataset_from_cfg(ds_cfg)
        current = getattr(self, "_prepared_key", None)
        desired = (target, float(ts))
        if current != desired:
            # Do not re-apply DP dynamically here
            self._prepare_splits(target_name=target, test_size=ts, dp_config=None)

    # --------------------------------------------------------------------------
    # Federated API
    # --------------------------------------------------------------------------
    def get_parameters(self, config):
        if self.model_type == "nn":
            return [val.cpu().numpy() for val in self.model.state_dict().values()]
        else:
            return booster_to_parameters(self.bst)

    def set_parameters(self, parameters):
        if self.model_type == "nn":
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)
        else:
            if parameters and len(parameters) > 0:
                self.bst = parameters_to_booster(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Allow server to override target/test_size dynamically
        self._ensure_prepared_from_config(config, dp_config=None)

        if self.model_type == "nn":
            self.model.train()
            local_epochs = config.get("local_epochs", 5)
            total_loss = 0.0
            for _ in range(local_epochs):
                for X_batch, y_batch in self.train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs.squeeze(), y_batch)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item() * X_batch.size(0)

            avg_loss = total_loss / (len(self.train_loader.dataset) * max(local_epochs, 1))
            with torch.no_grad():
                logits = self.model(self.X_tensor).squeeze()  # train set
                probs = torch.sigmoid(logits).cpu().numpy()
                y_true = self.y_tensor.cpu().numpy()
            th = config.get("threshold", 0.5)
            binary_preds = (probs >= th).astype(int)
            acc = accuracy_score(y_true, binary_preds)
            auc = roc_auc_score(y_true, probs)

            metrics = {
                "train_loss": avg_loss,
                "train_accuracy": acc,
                "train_auc": auc,
            }
            return self.get_parameters(config), len(self.X_tensor), metrics

        else:
            local_rounds = int(config.get("num_local_round", config.get("xgb_rounds", self.xgb_rounds)))
            self.bst = xgb.train(
                self.xgb_params,
                self.dtrain,
                num_boost_round=local_rounds,
                xgb_model=self.bst,  # continue from global
            )
            preds = self.bst.predict(self.dtrain)
            th = config.get("threshold", 0.5)
            binary_preds = (preds >= th).astype(int)
            acc = float((binary_preds == self.y_np).mean())
            auc = float(roc_auc_score(self.y_np, preds)) if len(np.unique(self.y_np)) > 1 else 0.0

            metrics = {"train_accuracy": acc, "train_auc": auc}
            return self.get_parameters(config), len(self.y_np), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Keep eval consistent with any overridden target/test_size
        self._ensure_prepared_from_config(config, dp_config=None)

        if self.model_type == "nn":
            self.model.eval()
            total_loss = 0.0
            all_probs, all_true = [], []
            with torch.no_grad():
                for X_batch, y_batch in self.test_loader:
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs.squeeze(), y_batch)
                    total_loss += loss.item() * X_batch.size(0)
                    probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                    all_probs.extend(probs.tolist())
                    all_true.extend(y_batch.cpu().numpy().tolist())

            avg_loss = total_loss / len(self.test_loader.dataset)
            th = config.get("threshold", 0.5)
            binary_preds = [1 if p >= th else 0 for p in all_probs]
            acc = accuracy_score(all_true, binary_preds)
            auc = roc_auc_score(all_true, all_probs)
            metrics = {"eval_loss": avg_loss, "eval_accuracy": acc, "eval_auc": auc}
            return float(avg_loss), len(self.test_loader.dataset), metrics

        else:
            th = config.get("threshold", 0.5)
            preds = self.bst.predict(self.dtest)
            binary = (preds >= th).astype(int)
            y_true = self.y_np_test
            acc = float((binary == y_true).mean())
            auc = float(roc_auc_score(y_true, preds)) if len(np.unique(y_true)) > 1 else 0.0
            metrics = {"eval_accuracy": acc, "eval_auc": auc}
            # loss optional for XGB; return 0.0 to satisfy Flower
            return 0.0, len(y_true), metrics

    def get_properties(self, config):
        hostname = socket.gethostname()
        os_type = platform.system()

        if self.model_type == "nn":
            num_samples = len(self.X_tensor)  # train samples
            num_features = self.X_tensor.shape[1]
        else:
            num_samples = len(self.y_np)      # train samples
            num_features = self.X_np.shape[1]

        features_str = ",".join(self.feature_names)
        classes_str = ",".join(map(str, self.classes))
        dist_str = ",".join(f"{cls}:{cnt}" for cls, cnt in self.label_counts.items())

        cpu_physical = psutil.cpu_count(logical=False)
        cpu_logical = psutil.cpu_count(logical=True)
        total_mem_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        driver_present = shutil.which('nvidia-smi') is not None
        gpu_count = 0
        if GPUtil and driver_present:
            try:
                gpu_count = len(GPUtil.getGPUs())
            except Exception:
                gpu_count = 0

        return {
            "hostname": hostname,
            "os_type": os_type,
            "num_samples": num_samples,
            "num_features": num_features,
            "features": features_str,
            "target": self.target_name,
            "classes": classes_str,
            "label_distribution": dist_str,
            "cpu_physical_cores": cpu_physical,
            "cpu_logical_cores": cpu_logical,
            "total_memory_gb": total_mem_gb,
            "gpu_driver_present": str(driver_present),
            "gpu_count": gpu_count,
            "model_type": self.model_type,
        }

    def start(self) -> None:
        fl.client.start_numpy_client(server_address=self.server_address, client=self)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower client with NN/XGBoost + optional DP for NN")
    parser.add_argument("--server_address", type=str, required=True, help="ex: 127.0.0.1:8080")
    parser.add_argument("--data_path", type=str, default="data/data.csv", help="CSV path")

    # Mode
    parser.add_argument("--model", type=str, default="nn", choices=["nn", "xgb"], help="Client model type")

    # DP (NN only)
    parser.add_argument("--dp", action="store_true", help="Activer la confidentialité différentielle (NN uniquement)")
    parser.add_argument("--noise_multiplier", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)

    # XGBoost params
    parser.add_argument("--xgb_eta", type=float, default=0.1)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--xgb_rounds", type=int, default=50)
    args = parser.parse_args()

    dp_config = None
    if args.dp and args.model == "nn":
        dp_config = DPConfig(
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            batch_size=args.batch_size,
        )

    xgb_params = None
    if args.model == "xgb":
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": args.xgb_eta,
            "max_depth": args.xgb_max_depth,
            "subsample": args.xgb_subsample,
            "colsample_bytree": args.xgb_colsample_bytree,
            "tree_method": "hist",
        }

    client = FlowerClient(
        server_address=args.server_address,
        data_path=args.data_path,
        dp_config=dp_config,
        model_type=args.model,
        xgb_params=xgb_params,
        xgb_rounds=args.xgb_rounds if args.model == "xgb" else 0,
    )
    client.start()
