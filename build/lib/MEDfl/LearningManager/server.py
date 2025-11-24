#!/usr/bin/env python3

import copy
from typing import Dict, Optional, Tuple

import flwr as fl
import torch

from .client import FlowerClient
from .federated_dataset import FederatedDataset
from .model import Model
from .strategy import Strategy


class FlowerServer:
    """
    A class representing the central server for Federated Learning using Flower.

    Attributes:
        global_model (Model): The global model that will be federated among clients.
        strategy (Strategy): The strategy used for federated learning, specifying communication and aggregation methods.
        num_rounds (int): The number of federated learning rounds to perform.
        num_clients (int): The number of clients participating in the federated learning process.
        fed_dataset (FederatedDataset): The federated dataset used for training and evaluation.
        diff_priv (bool): Whether differential privacy is used during the federated learning process.
        accuracies (List[float]): A list to store the accuracy of the global model during each round.
        losses (List[float]): A list to store the loss of the global model during each round.
        flower_clients (List[FlowerClient]): A list to store the FlowerClient objects representing individual clients.
     
    """

    def __init__(
        self,
        global_model: Model,
        strategy: Strategy,
        num_rounds: int,
        num_clients: int,
        fed_dataset: FederatedDataset,
        diff_privacy: bool = False,
        client_resources: Optional[Dict[str, float]] = {'num_cpus': 1, 'num_gpus': 0.0}
    ) -> None:
        """
        Initialize a FlowerServer object with the specified parameters.

        Args:
            global_model (Model): The global model that will be federated among clients.
            strategy (Strategy): The strategy used for federated learning, specifying communication and aggregation methods.
            num_rounds (int): The number of federated learning rounds to perform.
            num_clients (int): The number of clients participating in the federated learning process.
            fed_dataset (FederatedDataset): The federated dataset used for training and evaluation.
            diff_privacy (bool, optional): Whether differential privacy is used during the federated learning process.
                                           Default is False.
        """
        self.device = torch.device(
            f"cuda" if torch.cuda.is_available() else "cpu"
        )
        self.global_model = global_model
        self.params = global_model.get_parameters()
        self.global_model.model = global_model.model.to(self.device)
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.fed_dataset = fed_dataset
        self.strategy = strategy
        self.client_resources = client_resources
        setattr(
            self.strategy.strategy_object,
            "min_available_clients",
            self.num_clients,
        )
        setattr(
            self.strategy.strategy_object,
            "initial_parameters",
            fl.common.ndarrays_to_parameters(self.params),
        )
        setattr(self.strategy.strategy_object, "evaluate_fn", self.evaluate)
        self.fed_dataset = fed_dataset
        self.diff_priv = diff_privacy
        self.accuracies = []
        self.losses = []
        self.auc = []
        self.flower_clients = []
        self.validate()

    def validate(self) -> None:
        """Validate global_model, strategy, num_clients, num_rounds, fed_dataset, diff_privacy"""
        if not isinstance(self.global_model, Model):
            raise TypeError("global_model argument must be a Model instance")

        # if not isinstance(self.strategy, Strategy):
        #     print(self.strategy)
        #     print(isinstance(self.strategy, Strategy))
        #     raise TypeError("strategy argument must be a Strategy instance")

        if not isinstance(self.num_clients, int):
            raise TypeError("num_clients argument must be an int")

        if not isinstance(self.num_rounds, int):
            raise TypeError("num_rounds argument must be an int")

        if not isinstance(self.diff_priv, bool):
            raise TypeError("diff_priv argument must be a bool")

    def client_fn(self, cid) -> FlowerClient:
        """
        Return a FlowerClient object for a specific client ID.

        Args:
            cid: The client ID.

        Returns:
            FlowerClient: A FlowerClient object representing the individual client.
        """
        
        device = torch.device(
            f"cuda:{int(cid) % 4}" if torch.cuda.is_available() else "cpu"
        )
        client_model = copy.deepcopy(self.global_model)
      
        trainloader = self.fed_dataset.trainloaders[int(cid)]
        valloader = self.fed_dataset.valloaders[int(cid)]
        # this helps in making plots
        
        client = FlowerClient(
            cid, client_model, trainloader, valloader, self.diff_priv
        )
        self.flower_clients.append(client)
        return client

    def evaluate(
        self,
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """
        Evaluate the global model on the validation dataset and update the accuracies and losses.

        Args:
            server_round (int): The current round of the federated learning process.
            parameters (fl.common.NDArrays): The global model parameters.
            config (Dict[str, fl.common.Scalar]): Configuration dictionary.

        Returns:
            Optional[Tuple[float, Dict[str, fl.common.Scalar]]]: The evaluation loss and accuracy.
        """
        testloader = self.fed_dataset.valloaders[0]
        
        self.global_model.set_parameters(
            parameters
        )  # Update model with the latest parameters
        loss, accuracy ,auc = self.global_model.evaluate(testloader, self.device)
        self.auc.append(auc)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        
        return loss, {"accuracy": accuracy}

    def run(self) -> None:
        """
        Run the federated learning process using Flower simulation.

        Returns:
            History: The history of the accuracies and losses during the training of each node 
        """
         # Increase the object store memory to the minimum allowed value or higher
        ray_init_args = {"include_dashboard": False
                         , "object_store_memory": 78643200
                        }
        self.fed_dataset.eng = None
        
        history = fl.simulation.start_simulation(
            client_fn=self.client_fn,
            num_clients=self.num_clients,
            config=fl.server.ServerConfig(self.num_rounds),
            strategy=self.strategy.strategy_object,
            ray_init_args=ray_init_args,
            client_resources = self.client_resources
        )

        return history

