import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Net defines a simple feedforward neural network with two hidden layers

    """

    def __init__(self, input_dim):
        """
        Initialize the layers of the network.

        Args:
            input_dim (int): Number of input features.
        """
        super().__init__()
        # First fully connected layer: input_dim → 64
        self.fc1 = nn.Linear(input_dim, 64)
        # Second fully connected layer: 64 → 32
        self.fc2 = nn.Linear(64, 32)
        # Output layer: 32 → 1
        self.fc3 = nn.Linear(32, 1)
        # Dropout with 30% probability
        self.dropout = nn.Dropout(0.3)
        # Batch normalization layers for hidden layers
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(32)

    def forward(self, x):
        """
        Define the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 1).
        """
        # Hidden layer 1: linear → batchnorm → ReLU → dropout
        x = F.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        # Hidden layer 2: linear → batchnorm → ReLU → dropout
        x = F.relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        # Output layer: linear
        return self.fc3(x)  # raw logits for BCEWithLogitsLoss
