import torch
import torch.nn as nn
from sklearn.svm import SVC

class DynamicModel:
    """DynamicModel class for creating various types of neural network models."""

    # Create a binary classifier model
    @staticmethod
    def create_binary_classifier(input_dim, hidden_dims, output_dim, activation='relu', dropout_rate=0.0,
                                 batch_norm=False, use_gpu=False):
        """
        Creates a binary classifier neural network model with customizable architecture.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dims (List[int]): List of dimensions for hidden layers.
            output_dim (int): Dimension of the output (number of classes).
            activation (str, optional): Activation function for hidden layers. Default is 'relu'.
            dropout_rate (float, optional): Dropout rate for regularization. Default is 0.0 (no dropout).
            batch_norm (bool, optional): Whether to apply batch normalization. Default is False.
            use_gpu (bool, optional): Whether to use GPU acceleration. Default is False.

        Returns:
            torch.nn.Module: Created PyTorch model.
        """

        layers = []

        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[0]))
            else:
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))

            activation_layer = nn.ReLU() if activation == 'relu' else nn.Sigmoid()
            layers.append(activation_layer)

            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Sigmoid())

        model = nn.Sequential(*layers)

        if use_gpu:
            model = model.cuda()

        return model

    # Create a multi-class classifier model
    @staticmethod
    def create_multiclass_classifier(input_dim, hidden_dims, output_dim, activation='relu', dropout_rate=0.0,
                                     batch_norm=False, use_gpu=False):
        """
        Creates a multiclass classifier neural network model with customizable architecture.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dims (List[int]): List of dimensions for hidden layers.
            output_dim (int): Dimension of the output (number of classes).
            activation (str, optional): Activation function for hidden layers. Default is 'relu'.
            dropout_rate (float, optional): Dropout rate for regularization. Default is 0.0 (no dropout).
            batch_norm (bool, optional): Whether to apply batch normalization. Default is False.
            use_gpu (bool, optional): Whether to use GPU acceleration. Default is False.

        Returns:
            torch.nn.Module: Created PyTorch model.
        """
        layers = []

        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[0]))
            else:
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))

            activation_layer = nn.ReLU() if activation == 'relu' else nn.Sigmoid()
            layers.append(activation_layer)

            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.LogSoftmax(dim=1))

        model = nn.Sequential(*layers)

        if use_gpu:
            model = model.cuda()

        return model

    # Create a linear regressor model
    @staticmethod
    def create_linear_regressor(input_dim, output_dim, use_gpu=False):
        """
        Creates a linear regressor neural network model.

        Args:
            input_dim (int): Dimension of the input data.
            output_dim (int): Dimension of the output.

        Returns:
            torch.nn.Module: Created PyTorch model.
        """
        class LinearRegressionModel(nn.Module):
            def __init__(self):
                super(LinearRegressionModel, self).__init__()
                self.linear = nn.Linear(input_dim, output_dim)

            def forward(self, x):
                return self.linear(x)

        model = LinearRegressionModel()

        if use_gpu:
            model = model.cuda()

        return model

    # Create a logistic regressor model
    @staticmethod
    def create_logistic_regressor(input_dim, use_gpu=False):
        """
        Creates a logistic regressor neural network model.

        Args:
            input_dim (int): Dimension of the input data.

        Returns:
            torch.nn.Module: Created PyTorch model.
        """
        class LogisticRegressionModel(nn.Module):
            def __init__(self):
                super(LogisticRegressionModel, self).__init__()
                self.linear = nn.Linear(input_dim, 1)

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        model = LogisticRegressionModel()

        if use_gpu:
            model = model.cuda()

        return model
    
    @staticmethod
    def create_convolutional_neural_network(input_channels, output_channels, kernel_size, use_gpu=False):
        """
        Creates a convolutional neural network (CNN) model.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.

        Returns:
            torch.nn.Module: Created PyTorch model.
        """

        model = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        if use_gpu:
            model = model.cuda()

        return model
    
    @staticmethod
    def create_recurrent_neural_network(input_size, hidden_size, use_gpu=False):
        """
        Creates a recurrent neural network (RNN) model.

        Args:
            input_size (int): Size of the input.
            hidden_size (int): Size of the hidden layer.

        Returns:
            torch.nn.Module: Created PyTorch model.
        """

        model = nn.RNN(input_size, hidden_size, batch_first=True)

        if use_gpu:
            model = model.cuda()

        return model
    
    @staticmethod
    def create_lstm_network(input_size, hidden_size, use_gpu=False):
        """
        Creates a Long Short-Term Memory (LSTM) network model.

        Args:
            input_size (int): Size of the input layer.
            hidden_size (int): Size of the hidden layer.

        Returns:
            torch.nn.Module: Created PyTorch model.
        """

        model = nn.LSTM(input_size, hidden_size, batch_first=True)

        if use_gpu:
            model = model.cuda()

        return model

    # Create the dynamic model
    def create_model(self, model_type: str, params_dict={}) -> torch.nn.Module:
        """
        Create a specific type of model dynamically based on the given parameters.

        Args:
            model_type (str): Type of the model to create ('Binary Classifier', 'Multiclass Classifier', 'Linear Regressor', 'Logistic Regressor', 'SVM', 'Neural Network Classifier', 'Convolutional Neural Network', 'Recurrent Neural Network', 'LSTM Network', 'Autoencoder').
            params_dict (dict): Dictionary containing parameters for model creation.

        Returns:
            torch.nn.Module: Created PyTorch model.
        """
        if model_type == 'Binary Classifier':
            return self.create_binary_classifier(
                params_dict['input_dim'], params_dict['hidden_dims'],
                params_dict['output_dim'], params_dict.get('activation', 'relu'),
                params_dict.get('dropout_rate', 0.0), params_dict.get('batch_norm', False),
                params_dict.get('use_gpu', False)
            )
        elif model_type == 'Multiclass Classifier':
            return self.create_multiclass_classifier(
                params_dict['input_dim'], params_dict['hidden_dims'],
                params_dict['output_dim'], params_dict.get('activation', 'relu'),
                params_dict.get('dropout_rate', 0.0), params_dict.get('batch_norm', False),
                params_dict.get('use_gpu', False)
            )
        elif model_type == 'Linear Regressor':
            return self.create_linear_regressor(
                params_dict['input_dim'], params_dict['output_dim'],
                params_dict.get('use_gpu', False)
            )
        elif model_type == 'Logistic Regressor':
            return self.create_logistic_regressor(
                params_dict['input_dim'], params_dict.get('use_gpu', False)
            )
        elif model_type == 'Neural Network Classifier':
            return self.create_neural_network_classifier(
                params_dict['input_dim'], params_dict['output_dim'],
                params_dict['hidden_dims'], params_dict.get('activation', 'relu'),
                params_dict.get('dropout_rate', 0.0), params_dict.get('batch_norm', False),
                params_dict.get('num_layers', 2), params_dict.get('use_gpu', False)
            )
        elif model_type == 'Convolutional Neural Network':
            return self.create_convolutional_neural_network(
                params_dict['input_channels'], params_dict['output_channels'],
                params_dict['kernel_size'], params_dict.get('use_gpu', False)
            )
        elif model_type == 'Recurrent Neural Network':
            return self.create_recurrent_neural_network(
                params_dict['input_size'], params_dict['hidden_size'],
                params_dict.get('use_gpu', False)
            )
        elif model_type == 'LSTM Network':
            return self.create_lstm_network(
                params_dict['input_size'], params_dict['hidden_size'],
                params_dict.get('use_gpu', False)
            )
        elif model_type == 'Autoencoder':
            return self.create_autoencoder(
                params_dict['input_size'], params_dict['encoder_hidden_size'],
                params_dict.get('use_gpu', False)
            )
        else:
            raise ValueError("Invalid model type provided")



