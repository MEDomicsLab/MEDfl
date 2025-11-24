#!/usr/bin/env python3
import flwr as fl
from opacus import PrivacyEngine
from torch.utils.data import DataLoader

from .model import Model
from .utils import params
import torch

class FlowerClient(fl.client.NumPyClient):
    """
    FlowerClient class for creating MEDfl clients.

    Attributes:
        cid (str): Client ID.
        local_model (Model): Local model of the federated learning network.
        trainloader (DataLoader): DataLoader for training data.
        valloader (DataLoader): DataLoader for validation data.
        diff_priv (bool): Flag indicating whether to use differential privacy.
    """
    def __init__(self, cid: str, local_model: Model, trainloader: DataLoader, valloader: DataLoader, diff_priv: bool = params["diff_privacy"]):
        """
        Initializes the FlowerClient instance.

        Args:
            cid (str): Client ID.
            local_model (Model): Local model of the federated learning network.
            trainloader (DataLoader): DataLoader for training data.
            valloader (DataLoader): DataLoader for validation data.
            diff_priv (bool): Flag indicating whether to use differential privacy.
        """
        self.cid = cid
        self.local_model = local_model
        self.trainloader = trainloader
        self.valloader = valloader
        if torch.cuda.is_available():
            num_cuda_devices = torch.cuda.device_count()
            if num_cuda_devices > 0:
                device_idx = int(self.cid) % num_cuda_devices
                self.device = torch.device(f"cuda:{device_idx}")
                self.local_model.model.to(self.device)
            else:
                # Handle case where CUDA is available but no CUDA devices are found
                raise RuntimeError("CUDA is available, but no CUDA devices are found.")
        else:
            # Handle case where CUDA is not available
            self.device = torch.device("cpu")
            self.local_model.model.to(self.device)

        self.privacy_engine = PrivacyEngine(secure_mode=False)
        self.diff_priv = diff_priv
        self.epsilons = []
        self.accuracies = []
        self.losses = []
        if self.diff_priv:
            model, optimizer, self.trainloader = self.privacy_engine.make_private_with_epsilon(
                module=self.local_model.model.train(),
                optimizer=self.local_model.optimizer,
                data_loader=self.trainloader,
                epochs=params["train_epochs"],
                target_epsilon=float(params["EPSILON"]),
                target_delta= float(params["DELTA"]),
                max_grad_norm=params["MAX_GRAD_NORM"],
            )
            setattr(self.local_model, "model", model)
            setattr(self.local_model, "optimizer", optimizer)
        self.validate()

    def validate(self):
        """Validates cid, local_model, trainloader, valloader."""
        if not isinstance(self.cid, str):
            raise TypeError("cid argument must be a string")

        if not isinstance(self.local_model, Model):
            raise TypeError("local_model argument must be a MEDfl.LearningManager.model.Model")

        if not isinstance(self.trainloader, DataLoader):
            raise TypeError("trainloader argument must be a torch.utils.data.dataloader")

        if not isinstance(self.valloader, DataLoader):
            raise TypeError("valloader argument must be a torch.utils.data.dataloader")

        if not isinstance(self.diff_priv, bool):
            raise TypeError("diff_priv argument must be a bool")

    def get_parameters(self, config):
        """
        Returns the current parameters of the local model.

        Args:
            config: Configuration information.

        Returns:
            Numpy array: Parameters of the local model.
        """
        print(f"[Client {self.cid}] get_parameters")
        return self.local_model.get_parameters()

    def fit(self, parameters, config):
        """
        Fits the local model to the received parameters using federated learning.

        Args:
            parameters: Parameters received from the server.
            config: Configuration information.

        Returns:
            Tuple: Parameters of the local model, number of training examples, and privacy information.
        """
        print('\n -------------------------------- \n  this is the config of the client')
        print(f"[Client {self.cid}] fit, config: {config}")
        # print(config['epochs'])
        print('\n -------------------------------- \n  ')
        self.local_model.set_parameters(parameters)
        for _ in range(params["train_epochs"]):
            epsilon = self.local_model.train(
                self.trainloader,
                epoch=_,
                device=self.device,
                privacy_engine=self.privacy_engine,
                diff_priv=self.diff_priv,
            )
            self.epsilons.append(epsilon)
        print(f"epsilon of client {self.cid} : eps = {epsilon}")
        return (
            self.local_model.get_parameters(),
            len(self.trainloader),
            {"epsilon": epsilon},
        )

    def evaluate(self, parameters, config):
        """
        Evaluates the local model on the validation data and returns the loss and accuracy.

        Args:
            parameters: Parameters received from the server.
            config: Configuration information.

        Returns:
            Tuple: Loss, number of validation examples, and accuracy information.
        """
        print(f"[Client {self.cid}] evaluate, config: {config}")
        self.local_model.set_parameters(parameters)
        loss, accuracy , auc = self.local_model.evaluate(
            self.valloader, device=self.device
        )
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
