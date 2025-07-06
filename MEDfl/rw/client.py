import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from model import Net

# Dummy training data
X_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y_train = torch.tensor([[1.0], [0.0]])

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net()
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
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
        return 0.5, len(X_train), {}

fl.client.start_numpy_client(server_address="100.65.215.27:8080", client=FlowerClient())
