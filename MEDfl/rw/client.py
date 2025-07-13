import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from model import Net

# Dummy training data
X_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y_train = torch.tensor([[1.0], [0.0]])

class FlowerClient(fl.client.NumPyClient):
    """
    FlowerClient implements a federated learning client using Flower (flwr) 
    with a PyTorch model.

    Attributes:
        model (torch.nn.Module): The local neural network model.
        loss_fn (torch.nn.Module): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model weights.

    Methods:
        get_parameters(config):
            Retrieve the current model parameters as a list of NumPy arrays 
            to send to the server.
            
        set_parameters(parameters):
            Update the local model with parameters received from the server.
            
        fit(parameters, config):
            Perform local training on the client using the received parameters. 
            Trains for a fixed number of epochs and returns updated parameters, 
            the number of examples used, and an optional metrics dictionary.
            
        evaluate(parameters, config):
            Evaluate the model on local data after receiving the global parameters 
            from the server. Returns loss, the number of examples used, and an 
            optional metrics dictionary.

    """

    def __init__(self):
        """Initialize the client with the model, loss function, and optimizer."""
        self.model = Net()
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        """
        Retrieve the current model parameters.

        Args:
            config (dict): Configuration dictionary from the server.

        Returns:
            List of NumPy arrays representing the model parameters.
        """
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """
        Update the local model with parameters received from the server.

        Args:
            parameters (list): List of NumPy arrays representing the model parameters.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Perform local training on the client.

        Args:
            parameters (list): Parameters received from the server.
            config (dict): Configuration dictionary from the server.

        Returns:
            tuple: Updated parameters, number of examples used, and metrics dict.
        """
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(5):
            self.optimizer.zero_grad()
            output = self.model(X_train)
            loss = self.loss_fn(output, y_train)
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(config), len(X_train), {}

    def evaluate(self, parameters, config):
        """
        Evaluate the model on local data.

        Args:
            parameters (list): Parameters received from the server.
            config (dict): Configuration dictionary from the server.

        Returns:
            tuple: Evaluation loss, number of examples used, and metrics dict.
        """
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_train)
            loss = self.loss_fn(output, y_train).item()
        return loss, len(X_train), {}

# Start the Flower client
fl.client.start_numpy_client(server_address="100.65.215.27:8080", client=FlowerClient())
