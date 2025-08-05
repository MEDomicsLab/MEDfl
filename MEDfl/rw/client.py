# client.py

import argparse
import pandas as pd
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from MEDfl.rw.model import Net  # votre définition de modèle
import socket
import platform
import psutil
import shutil

try:
    import GPUtil
except ImportError:
    GPUtil = None


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


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        server_address: str,
        data_path: str = "data/data.csv",
        dp_config: DPConfig = None,
    ):
        self.server_address = server_address

        # 1) Chargement des données
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Métadonnées pour get_properties()
        self.feature_names = df.columns[:-1].tolist()
        self.target_name = df.columns[-1]
        self.label_counts = df[self.target_name].value_counts().to_dict()
        self.classes = sorted(self.label_counts.keys())

        # Tenseurs pour les métriques
        self.X_tensor = torch.tensor(X, dtype=torch.float32)
        self.y_tensor = torch.tensor(y, dtype=torch.float32)

        # 2) DataLoader
        batch_size = dp_config.batch_size if dp_config else 32
        dataset = TensorDataset(self.X_tensor, self.y_tensor)
        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 3) Modèle, critère, optimiseur
        input_dim = X.shape[1]
        self.model = Net(input_dim)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        # 4) Configuration DP (Opacus)
        self.privacy_engine = None
        if dp_config:
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

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        local_epochs = config.get("local_epochs", 5)
        total_loss = 0.0
        print(f"Training for {local_epochs} epochs...")

        for epoch in range(local_epochs):
            print(f"Epoch {epoch + 1}/{local_epochs}")
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / (len(self.train_loader.dataset) * local_epochs)

        # Compute metrics after last epoch
        with torch.no_grad():
            logits = self.model(self.X_tensor).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true = self.y_tensor.cpu().numpy()
        binary_preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_true, binary_preds)
        auc = roc_auc_score(y_true, probs)

        hostname = socket.gethostname()
        os_type = platform.system()
        metrics = {
            "hostname": hostname,
            "os_type": os_type,
            "train_loss": avg_loss,
            "train_accuracy": acc,
            "train_auc": auc,
        }

        return self.get_parameters(config), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        all_probs, all_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in self.train_loader:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                total_loss += loss.item() * X_batch.size(0)
                probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_true.extend(y_batch.cpu().numpy().tolist())

        avg_loss = total_loss / len(self.train_loader.dataset)
        binary_preds = [1 if p >= 0.5 else 0 for p in all_probs]
        acc = accuracy_score(all_true, binary_preds)
        auc = roc_auc_score(all_true, all_probs)

        metrics = {
            "eval_loss": avg_loss,
            "eval_accuracy": acc,
            "eval_auc": auc,
        }
        print(f"Evaluation metrics: {metrics}")

        return float(avg_loss), len(self.train_loader.dataset), metrics

    def get_properties(self, config):
        """
        Return dataset statistics before training starts.
        Only scalar values allowed (int, float, str, bool, bytes).
        """
        hostname = socket.gethostname()
        os_type = platform.system()
        num_samples = len(self.X_tensor)
        num_features = self.X_tensor.shape[1]

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
        }

    def start(self) -> None:
        """
        Start this client against its configured server_address.
        Blocks until the server shuts down.
        """
        fl.client.start_numpy_client(
            server_address=self.server_address,
            client=self,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flower client with DP support and metrics"
    )
    parser.add_argument(
        "--server_address",
        type=str,
        required=True,
        help="Adresse du serveur Flower (ex : 127.0.0.1:8080)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/data.csv",
        help="Chemin vers les données CSV",
    )
    parser.add_argument(
        "--dp",
        action="store_true",
        help="Activer la confidentialité différentielle",
    )
    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=1.0,
        help="Noise multiplier pour DP",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Norme max pour le clipping",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Taille du batch",
    )
    args = parser.parse_args()

    dp_config = None
    if args.dp:
        dp_config = DPConfig(
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            batch_size=args.batch_size,
        )

    client = FlowerClient(
        server_address=args.server_address,
        data_path=args.data_path,
        dp_config=dp_config,
    )
    client.start()
