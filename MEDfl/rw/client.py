# client.py
"""
Federated Learning Client with Optional Differential Privacy.

"""

import argparse
import pandas as pd
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from model import Net  # Local model definition
import socket
import platform


class DPConfig:
    """
    Configuration for Differential Privacy (DP) settings.

    Attributes:
        noise_multiplier (float): Noise multiplier for DP.
        max_grad_norm (float): Maximum gradient norm (clipping threshold).
        batch_size (int): Batch size for training.
        secure_rng (bool): Whether to use a secure RNG for DP noise.
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
    """
    FlowerClient: A federated learning client that trains a PyTorch model
    and optionally applies differential privacy.

    """

    def __init__(
        self,
        server_address: str,
        data_path: str = "data/data.csv",
        dp_config: DPConfig = None,
    ):
        """
        Initialize client by loading data, creating model, optimizer,
        and optionally enabling DP.

        Args:
            server_address (str): Flower server address.
            data_path (str): Path to CSV dataset.
            dp_config (DPConfig): Optional DP configuration.
        """
        self.server_address = server_address

        # ---------- Load Data ----------
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        self.X_tensor = torch.tensor(X, dtype=torch.float32)
        self.y_tensor = torch.tensor(y, dtype=torch.float32)

        batch_size = dp_config.batch_size if dp_config else 32
        dataset = TensorDataset(self.X_tensor, self.y_tensor)
        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # ---------- Model and Optimizer ----------
        input_dim = X.shape[1]
        self.model = Net(input_dim)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        # ---------- Differential Privacy ----------
        self.privacy_engine = None
        if dp_config is not None:
            try:
                from opacus import PrivacyEngine

                self.privacy_engine = PrivacyEngine()
                self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=dp_config.noise_multiplier,
                    max_grad_norm=dp_config.max_grad_norm,
                    secure_rng=dp_config.secure_rng,
                )
            except ImportError:
                print("⚠️ Opacus not installed, running without DP.")

    def get_parameters(self, config):
        """
        Get model parameters as a list of NumPy arrays.
        """
        return [val.cpu().numpy() for val in self.mo]()
