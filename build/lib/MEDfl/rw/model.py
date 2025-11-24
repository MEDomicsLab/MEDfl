import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "silu": nn.SiLU,
    "identity": nn.Identity,
}

def _make_activation(name):
    if name not in _ACTS:
        raise ValueError(f"Unsupported activation '{name}'. Choose from {list(_ACTS)}")
    return _ACTS[name]()

class Net(nn.Module):
    """
    Flexible MLP with task-aware output.
    - hidden_dims: e.g., [128, 64, 32]
    - activation: 'relu' | 'gelu' | 'leaky_relu' | ...
    - batchnorm: apply BatchNorm1d after each linear (except output)
    - dropout: float in [0,1] or list per hidden layer
    - task: 'binary' | 'multiclass' | 'multilabel' | 'regression'
    - num_classes: required for 'multiclass'; for 'binary' ignore; for 'multilabel' set to label count
    - return_logits: always True for training (recommended). Use .predict() for post-activation outputs.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=[64, 32],
        activation="relu",
        batchnorm=True,
        dropout=0.0,
        task="binary",
        num_classes=None,
        output_bias=True,
        return_logits=True,
        weight_init="kaiming",
    ):
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.return_logits = return_logits
        self.act = activation

        if task == "multiclass":
            if not num_classes or num_classes < 2:
                raise ValueError("For 'multiclass', num_classes >= 2 is required.")
            output_dim = num_classes
        elif task == "binary":
            output_dim = 1
        elif task == "multilabel":
            if not num_classes or num_classes < 1:
                raise ValueError("For 'multilabel', set num_classes = number of labels.")
            output_dim = num_classes
        elif task == "regression":
            output_dim = 1
        else:
            raise ValueError(f"Unknown task: {task}")

        # Normalize dropout to list per hidden layer
        if isinstance(dropout, (int, float)):
            dropout = [float(dropout)] * len(hidden_dims)
        elif dropout is None:
            dropout = [0.0] * len(hidden_dims)
        elif isinstance(dropout, list):
            if len(dropout) != len(hidden_dims):
                raise ValueError("Length of dropout list must match hidden_dims.")
        else:
            raise ValueError("dropout must be float|list[float]|None")

        self.blocks = nn.ModuleList()
        in_dim = input_dim
        for h, p in zip(hidden_dims, dropout):
            block = nn.ModuleDict({
                "lin": nn.Linear(in_dim, h, bias=True),
                "bn": nn.BatchNorm1d(h) if batchnorm else nn.Identity(),
                "act": _make_activation(activation),
                "drop": nn.Dropout(p) if p and p > 0 else nn.Identity(),
            })
            self.blocks.append(block)
            in_dim = h

        self.out = nn.Linear(in_dim, output_dim, bias=output_bias)

        # Weight init
        if weight_init != "none":
            self.apply(lambda m: self._init_weights(m, scheme=weight_init, activation=activation))

    @staticmethod
    def _init_weights(m, scheme, activation):
        if isinstance(m, nn.Linear):
            if scheme == "kaiming":
                nonlinearity = "leaky_relu" if activation == "leaky_relu" else "relu"
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=math.sqrt(5) if activation == "leaky_relu" else 0,
                    nonlinearity=nonlinearity,
                )
            elif scheme == "xavier":
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        for blk in self.blocks:
            x = blk["lin"](x)
            x = blk["bn"](x)
            x = blk["act"](x)
            x = blk["drop"](x)
        logits = self.out(x)
        return logits if self.return_logits else self._apply_output_activation(logits)

    # Inference helpers
    def _apply_output_activation(self, logits):
        if self.task == "binary":
            return torch.sigmoid(logits)               # (B, 1)
        elif self.task == "multiclass":
            return F.softmax(logits, dim=-1)           # (B, C)
        elif self.task == "multilabel":
            return torch.sigmoid(logits)               # (B, L)
        elif self.task == "regression":
            return logits                              # raw regression output
        else:
            raise RuntimeError("Invalid task")

    @torch.no_grad()
    def predict(self, x):
        out = self.forward(x)
        probs = self._apply_output_activation(out) if self.return_logits else out

        if self.task == "binary":
            return (probs >= 0.5).long()
        elif self.task == "multiclass":
            return probs.argmax(dim=-1)
        elif self.task == "multilabel":
            return (probs >= 0.5).long()
        elif self.task == "regression":
            return probs
