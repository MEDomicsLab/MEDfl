
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np

import optuna




class  Strategy:
    """
    A class representing a strategy for Federated Learning.

    Attributes:
        name (str): The name of the strategy. Default is "FedAvg".
        fraction_fit (float): Fraction of clients to use for training during each round. Default is 1.0.
        fraction_evaluate (float): Fraction of clients to use for evaluation during each round. Default is 1.0.
        min_fit_clients (int): Minimum number of clients to use for training during each round. Default is 2.
        min_evaluate_clients (int): Minimum number of clients to use for evaluation during each round. Default is 2.
        min_available_clients (int): Minimum number of available clients required to start a round. Default is 2.
        initial_parameters (Optional[]): The initial parameters of the server model 
    Methods:
     
    """

    def __init__(
        self,
        name: str = "FedAvg",
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters = [],
        evaluation_methode = "centralized"
    ) -> None:
        """
        Initialize a Strategy object with the specified parameters.

        Args:
            name (str): The name of the strategy. Default is "FedAvg".
            fraction_fit (float): Fraction of clients to use for training during each round. Default is 1.0.
            fraction_evaluate (float): Fraction of clients to use for evaluation during each round. Default is 1.0.
            min_fit_clients (int): Minimum number of clients to use for training during each round. Default is 2.
            min_evaluate_clients (int): Minimum number of clients to use for evaluation during each round. Default is 2.
            min_available_clients (int): Minimum number of available clients required to start a round. Default is 2.
            initial_parameters (Optional[]): The initial parametres of the server model 
            evaluation_methode ( "centralized" | "distributed")
        """
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.evaluate_fn = None
        self.name = name
    
    def optuna_fed_optimization(self, direction:str , hpo_rate:int , params_config):
        self.study = optuna.create_study(direction=direction)
        self.hpo_rate = hpo_rate
        self.params_config = params_config
    
 
    def create_strategy(self):
        self.strategy_object = self.get_strategy_by_name()(
            fraction_fit=self.fraction_fit,
            fraction_evaluate=self.fraction_evaluate,
            min_fit_clients=self.min_fit_clients,
            min_evaluate_clients=self.min_evaluate_clients,
            min_available_clients=self.min_available_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(self.initial_parameters),
            evaluate_fn=self.evaluate_fn
        )
    def get_strategy_by_name(self):
        return eval(f"fl.server.strategy.{self.name}")
    
    

