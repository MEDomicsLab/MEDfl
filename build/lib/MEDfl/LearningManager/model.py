#!/usr/bin/env python3
# froked from https://github.com/pythonlessons/mltu/blob/main/mltu/torch/model.py

import typing
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score,roc_auc_score

from .utils import params


class Model:
    """
    Model class for training and testing PyTorch neural networks.

    Attributes:
        model (torch.nn.Module): PyTorch neural network.
        optimizer (torch.optim.Optimizer): PyTorch optimizer.
        criterion (typing.Callable): Loss function.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: typing.Callable,
    ) -> None:
        """
        Initialize Model class with the specified model, optimizer, and criterion.

        Args:
            model (torch.nn.Module): PyTorch neural network.
            optimizer (torch.optim.Optimizer): PyTorch optimizer.
            criterion (typing.Callable): Loss function.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        # Get device on which model is running
        self.validate()

    def validate(self) -> None:
        """
        Validate model and optimizer.
        """
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError("model argument must be a torch.nn.Module")

        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError(
                "optimizer argument must be a torch.optim.Optimizer"
            )

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get the parameters of the model as a list of NumPy arrays.

        Returns:
            List[np.ndarray]: The parameters of the model as a list of NumPy arrays.
        """
        return [
            val.cpu().numpy() for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set the parameters of the model from a list of NumPy arrays.

        Args:
            parameters (List[np.ndarray]): The parameters to be set.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def train(
        self, train_loader, epoch, device, privacy_engine, diff_priv=False
    ) -> float:
        """
        Train the model on the given train_loader for one epoch.

        Args:
            train_loader: The data loader for training data.
            epoch (int): The current epoch number.
            device: The device on which to perform the training.
            privacy_engine: The privacy engine used for differential privacy (if enabled).
            diff_priv (bool, optional): Whether differential privacy is used. Default is False.

        Returns:
            float: The value of epsilon used in differential privacy.
        """
        self.model.train()
        epsilon = 0
        losses = []
        top1_acc = []

        for i, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.to(device), y_train.to(device)  
            
            self.optimizer.zero_grad()

            # compute output
            y_hat = torch.squeeze(self.model(X_train), 1)
            loss = self.criterion(y_hat, y_train)

            preds = np.argmax(y_hat.detach().cpu().numpy(), axis=0)
            labels = y_train.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = (preds == labels).mean()

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            self.optimizer.step()

            if diff_priv:
                epsilon = privacy_engine.get_epsilon(float(params["DELTA"]))

            if (i + 1) % 10 == 0:
                if diff_priv:
                    epsilon = privacy_engine.get_epsilon(float(params["DELTA"]))
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                        f"(ε = {epsilon:.2f}, δ = {params['DELTA']})"
                    )
                else:
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top1_acc) * 100:.6f}"
                    )

        return epsilon

    def evaluate(self, val_loader, device=torch.device("cpu")) -> Tuple[float, float]:
        """
        Evaluate the model on the given validation data.

        Args:
            val_loader: The data loader for validation data.
            device: The device on which to perform the evaluation. Default is 'cpu'.

        Returns:
            Tuple[float, float]: The evaluation loss and accuracy.
        """
        correct, total, loss, accuracy, auc = 0, 0, 0.0, [], []
        self.model.eval()

        with torch.no_grad():
            for X_test, y_test in val_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)  # Move data to device

                y_hat = torch.squeeze(self.model(X_test), 1)
               
                
                criterion = self.criterion.to(y_hat.device)
                loss += criterion(y_hat, y_test).item()


                # Move y_hat to CPU for accuracy computation
                y_hat_cpu = y_hat.cpu().detach().numpy()
                accuracy.append(accuracy_score(y_test.cpu().numpy(), y_hat_cpu.round()))

                # Move y_test to CPU for AUC computation
                y_test_cpu = y_test.cpu().numpy()
                y_prob_cpu = y_hat.cpu().detach().numpy()
                if (len(np.unique(y_test_cpu)) != 1):
                    auc.append(roc_auc_score(y_test_cpu, y_prob_cpu))

                total += y_test.size(0)
                correct += np.sum(y_hat_cpu.round() == y_test_cpu)

        loss /= len(val_loader.dataset)
        return loss, np.mean(accuracy), np.mean(auc)


    @staticmethod
    def save_model(model , model_name:str):
        """
        Saves a PyTorch model to a file.

        Args:
            model (torch.nn.Module): PyTorch model to be saved.
            model_name (str): Name of the model file.

        Raises:
            Exception: If there is an issue during the saving process.

        Returns:
            None
        """
        try:
            torch.save(model, '../../notebooks/.ipynb_checkpoints/trainedModels/' + model_name + ".pth")
        except Exception as e:
            raise Exception(f"Error saving the model: {str(e)}")
    
    @staticmethod
    def load_model(model_path: str):
        """
        Loads a PyTorch model from a file.

        Args:
            model_path (str): Path to the model file to be loaded.

        Returns:
            torch.nn.Module: Loaded PyTorch model.
        """
        # Ensure models are loaded onto the CPU when CUDA is not available
        torch_kwargs = {"weights_only": False}
        if torch.cuda.is_available():
            loaded_model = torch.load(model_path , **torch_kwargs)
        else:
            loaded_model = torch.load(model_path, map_location=torch.device('cpu') , **torch_kwargs)
        return loaded_model


