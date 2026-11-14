<div align="center">

<img src="https://raw.githubusercontent.com/MEDomics-UdeS/MEDfl/main/medfl_logo.png" alt="MEDfl logo" width="220"/>

# MEDfl

**A collaborative framework for privacy-preserving federated learning in medicine**

[![PyPI version](https://img.shields.io/pypi/v/MEDfl)](https://pypi.org/project/medfl/)
[![Python Versions](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/MEDfl)](https://pypi.org/project/medfl/)
[![Contributors](https://img.shields.io/github/contributors/MEDomics-UdeS/MEDfl)](https://github.com/MEDomics-UdeS/MEDfl/graphs/contributors)

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Modes of Operation](#modes-of-operation)
5. [Quick Start — Real-World Federated Learning](#quick-start--real-world-federated-learning)
6. [Federated XGBoost](#federated-xgboost)
7. [Federated Explainability (SHAP)](#federated-explainability-shap)
8. [Federated Holdout Testing & Threshold Optimization](#federated-holdout-testing--threshold-optimization)
9. [Simulation Mode](#simulation-mode)
10. [Package Structure](#package-structure)
11. [Documentation](#documentation)
12. [Testing](#testing)
13. [Contributing](#contributing)
14. [Acknowledgment](#acknowledgment)
15. [Authors](#authors)
16. [License](#license)

---

## Overview

**MEDfl** is an open-source Python framework for **Federated Learning (FL)** in the **medical and healthcare domain**. It lets several institutions (hospitals, research centers, clinics) train a shared model **without ever moving patient data off-site**: only model updates and aggregate statistics leave each site.

MEDfl is built on top of [Flower](https://flower.ai) and supports both:

- **Simulation** of federated experiments on a single machine, for benchmarking and research; and
- **Real-world deployment** across physically distributed client machines.

Beyond training, MEDfl covers the full clinical-ML workflow: **differential privacy**, **gradient-boosted trees (XGBoost)**, **federated SHAP explainability**, **federated holdout evaluation** with pooled ROC/AUC, and **decision-threshold optimization** — all computed without centralizing data.

---

## Key Features

| Area | Capability |
| --- | --- |
| 🧩 **Two operation modes** | Local **simulation** (Flower simulation engine) and **real-world** distributed deployment (Flower server/clients). |
| 🧠 **Neural networks** | Configurable MLP (`Net`) with binary, multiclass, multilabel and regression heads; `DynamicModel` factory in simulation mode (binary/multiclass classifiers, linear & logistic regressors, CNN, RNN, LSTM, autoencoder). |
| 🌲 **Federated XGBoost** | Bagging-based federated XGBoost for binary classification, multiclass classification and regression, in both modes. |
| 🔁 **Aggregation strategies** | Any Flower strategy by name (`FedAvg`, `FedAdam`, `FedYogi`, `FedProx`, …) plus `FedXgbBagging` for trees. |
| 🔒 **Differential privacy** | Client-side DP-SGD via [Opacus](https://opacus.ai) (`DPConfig`: noise multiplier, gradient clipping, secure RNG). |
| 🔍 **Federated SHAP** | Each client computes local SHAP values (Gradient, Deep or Tree explainer); the server aggregates only summary statistics into global feature importance. |
| 🧪 **Federated holdout testing** | Final global model is tested on every client's private holdout file; the server reconstructs **exact pooled** confusion-matrix metrics and a pooled ROC curve/AUC. |
| 🎯 **Threshold optimization** | Optimized decision threshold (Youden's J or target sensitivity) chosen on pooled learning-set score histograms, then validated on the holdout. |
| ✂️ **Flexible data splits** | Global or per-client validation/test fractions, or explicit test IDs per client. |
| 🗄️ **Experiment management** | Simulation-mode networks, nodes, datasets, setups, pipelines and results stored in a relational database (SQLite / MySQL). |
| ⚙️ **Hyperparameter optimization** | Grid search and [Optuna](https://optuna.org)-based (including federated) hyperparameter tuning. |
| 📊 **Visualization** | Accuracy/loss curves, global and per-node confusion matrices, classification reports. |
| 💾 **Checkpointing** | Global model saved every *N* rounds (`.npz` for neural networks, `.ubj` for XGBoost). |

---

## Installation

MEDfl requires **Python 3.9 or later**.

**From PyPI**

```bash
pip install medfl
```

**From source (development version)**

```bash
git clone https://github.com/MEDomics-UdeS/MEDfl.git
cd MEDfl
pip install -e .
```

Main dependencies: `flwr[simulation]`, `torch`, `opacus`, `xgboost`, `shap`, `scikit-learn`, `pandas`, `optuna`, `SQLAlchemy`. See [requirements.txt](requirements.txt) for exact versions.

---

## Modes of Operation

| Mode | Package | Description | Typical use case |
| --- | --- | --- | --- |
| **Simulation** | `MEDfl.NetManager`, `MEDfl.LearningManager` | All clients run locally in a controlled environment; data partitions and experiments are tracked in a database. | Benchmarking, prototyping, methodological research. |
| **Real-world** | `MEDfl.rw` | A central server and remote clients communicate over the network; each client reads its own local CSV. | Multi-institution collaborations, production deployments. |

---

## Quick Start — Real-World Federated Learning

In real-world mode, **the server owns the experiment configuration** (features, target, splits, rounds, hyperparameters) and pushes it to the clients. Clients only need to point to their local CSV file.

### 1. Server

```python
from MEDfl.rw.server import FederatedServer
from MEDfl.rw.strategy import Strategy

strategy = Strategy(
    name="FedAvg",                   # any Flower strategy: FedAvg, FedAdam, FedYogi, FedProx, ...
    fraction_fit=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    local_epochs=3,
    threshold=0.5,
    features="age,bmi,glucose,blood_pressure",  # comma-separated input columns
    target="label",
    val_fraction=0.10,
    test_fraction=0.10,
    savingPath="./checkpoints",
    saveOnRounds=5,
)

server = FederatedServer(
    host="0.0.0.0",
    port=8080,
    num_rounds=10,
    strategy=strategy,
)
server.start()
```

### 2. Client (run on each participating site)

```python
from MEDfl.rw.client import FlowerClient, DPConfig

client = FlowerClient(
    server_address="SERVER_IP:8080",
    data_path="data/site_data.csv",
    dp_config=DPConfig(                # optional — omit to disable differential privacy
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        batch_size=32,
    ),
)
client.start()
```

### Per-client data splits

By default every client uses the global `val_fraction` / `test_fraction`. With `split_mode="per_client"`, each client (identified by its hostname, or by the `MEDFL_AGENT_ID` environment variable when set) can have its own fractions or a fixed list of test IDs:

```python
strategy = Strategy(
    ...,
    split_mode="per_client",
    id_col="patient_id",
    client_fractions={
        "hospital-a": {"val_fraction": 0.15, "test_fraction": 0.20},
        "hospital-b": {"test_ids": "1042,1043,1077"},
    },
)
```

> 💡 **Networking tip:** for deployments across institutions, a mesh VPN such as [Tailscale](https://tailscale.com) is a simple way to put the server and clients on the same secure private network. TLS certificates can also be passed to `FederatedServer(certificates=...)`.

---

## Federated XGBoost

MEDfl supports **bagging-based federated XGBoost** (`FedXgbBagging`): each client boosts locally from the current global model, and the server combines the resulting trees.

**Server**

```python
from MEDfl.rw.server import FederatedServer
from MEDfl.rw.xgboost import XGBoostStrategy

strategy = XGBoostStrategy(
    task="binary",                     # "binary", "multiclass" or "regression"
    xgb_params={"max_depth": 6, "eta": 0.1, "tree_method": "hist"},
    local_num_boost_round=10,
    features="age,bmi,glucose,blood_pressure",
    target="label",
    min_fit_clients=2,
    min_available_clients=2,
    savingPath="./checkpoints",        # models saved as .ubj
)

FederatedServer(host="0.0.0.0", port=8080, num_rounds=10, strategy=strategy).start()
```

**Client**

```python
from MEDfl.rw.xgboost import XGBoostFlowerClient

XGBoostFlowerClient(
    server_address="SERVER_IP:8080",
    data_path="data/site_data.csv",
).start()
```

Sensible defaults (objective, evaluation metric, `hist` tree method) are chosen per task; any value in `xgb_params` overrides them. For multiclass, set `num_classes`.

In simulation mode, the equivalent components are `XGBoostSimulationStrategy`, `XGBoostSimulationServer` and `XGBoostSimulationClient` in `MEDfl.LearningManager.xgboost`.

---

## Federated Explainability (SHAP)

After the final round, MEDfl can compute **global feature importance without sharing raw SHAP values**. Each client explains the final global model on its own data and returns only aggregate statistics (sums, squared sums, sign counts); the server combines them into federated mean |SHAP|, mean signed SHAP and dispersion per feature.

```python
from MEDfl.LearningManager.shap import SHAPConfig

shap_config = SHAPConfig(
    enabled=True,
    explainer="gradient",          # "gradient" or "deep" for neural networks, "tree" for XGBoost
    data_split="validation",       # "train", "validation" or "test"
    background_size=100,
    explanation_size=500,
    minimum_samples=10,            # clients with fewer samples are skipped
    model_output="raw",            # XGBoost only: "raw" (log-odds) or "probability"
)

server = FederatedServer(..., strategy=strategy, shap_config=shap_config)
server.start()

print(server.federated_shap_result)   # aggregated global importance
print(server.local_shap_results)      # optional per-client summaries
```

| Backend | Supported explainers | Supported tasks |
| --- | --- | --- |
| PyTorch | `gradient`, `deep` | binary classification and other single-output models |
| XGBoost | `tree` | binary classification, regression |

The same `SHAPConfig` can be passed to the simulation-mode `FlowerServer` and `XGBoostSimulationServer`.

---

## Federated Holdout Testing & Threshold Optimization

Clinical models need an **unbiased final evaluation** on data never used for training. With a federated holdout test, each client keeps a private holdout CSV; after the last round the final global model is evaluated on it, and clients return only:

- the confusion matrix at the decision threshold,
- their local metrics, and
- two histograms of predicted probabilities (positives / negatives).

The server sums confusion matrices to obtain **exact pooled** sensitivity, specificity, PPV, NPV, F1 and accuracy, and sums histograms to rebuild a **pooled ROC curve and AUC** (exact up to the histogram bin width).

```python
from MEDfl.rw.holdout import HoldoutConfig

holdout_config = HoldoutConfig(
    enabled=True,
    threshold=0.5,                 # used when threshold_source="manual"
    num_bins=100,                  # ROC resolution
    minimum_samples=10,
    threshold_source="optimized",  # "manual" or "optimized"
    threshold_criterion="youden",  # "youden" or "target_sensitivity"
    target_sensitivity=0.9,        # used with "target_sensitivity"
)

server = FederatedServer(..., strategy=strategy, holdout_config=holdout_config)
server.start()
print(server.federated_holdout_result)
```

On each client, point to the holdout file:

```python
FlowerClient(server_address="SERVER_IP:8080",
             data_path="data/site_data.csv",
             holdout_path="data/site_holdout.csv").start()
```

**Optimized threshold.** With `threshold_source="optimized"`, clients also send score histograms of their whole learning set (train + validation + test). The server selects the threshold on these pooled histograms (Youden's J or the lowest threshold reaching a target sensitivity), then reports holdout metrics at that threshold. The holdout data is **never** used to choose the threshold, so holdout results remain an unbiased estimate.

Holdout testing works for both the PyTorch and the XGBoost real-world backends (binary classification).

---

## Simulation Mode

Simulation mode is designed for research and benchmarking. Networks, hospital nodes, datasets, FL setups, pipelines and results are persisted in a database, making experiments reproducible and comparable.

```python
import torch.nn as nn
import torch.optim as optim

from MEDfl.LearningManager.utils import set_db_config
from MEDfl.NetManager.database_connector import DatabaseManager
from MEDfl.NetManager.network import Network
from MEDfl.NetManager.node import Node
from MEDfl.NetManager.flsetup import FLsetup
from MEDfl.LearningManager.dynamicModal import DynamicModel
from MEDfl.LearningManager.model import Model
from MEDfl.LearningManager.strategy import Strategy
from MEDfl.LearningManager.server import FlowerServer
from MEDfl.LearningManager.flpipeline import FLpipeline

# 1. Database
set_db_config("./medfl_database.db")
db = DatabaseManager()
db.connect()
db.create_MEDfl_db(path_to_csv="data/master_dataset.csv")

# 2. Network, setup and nodes
net = Network("Net1")
net.create_network()
net.create_master_dataset("data/master_dataset.csv")

fl_setup = FLsetup(name="setup_1", description="Diabetes study", network=net)
fl_setup.create()

for name, path, train in [("hospital_1", "data/h1.csv", 1),
                          ("hospital_2", "data/h2.csv", 1),
                          ("hospital_3", "data/h3.csv", 0)]:   # train=0 → test-only node
    node = Node(name=name, train=train)
    net.add_node(node)
    node.upload_dataset(name, path)

fl_dataset = fl_setup.create_federated_dataset(output="diabetes", fit_encode=[], to_drop=["diabetes"])

# 3. Model and strategy
torch_model = DynamicModel().create_model(
    model_type="Binary Classifier",
    params_dict={"input_dim": 8, "output_dim": 1, "hidden_dims": [16, 32]},
)
global_model = Model(torch_model, optim.SGD(torch_model.parameters(), lr=1e-3), nn.BCELoss())

strategy = Strategy("FedAdam", min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
                    initial_parameters=global_model.get_parameters())
strategy.create_strategy()

# 4. Train and evaluate
server = FlowerServer(global_model, strategy=strategy, num_rounds=10,
                      num_clients=len(fl_dataset.trainloaders), fed_dataset=fl_dataset,
                      diff_privacy=False)
pipeline = FLpipeline(name="pipeline_1", description="First FL pipeline", server=server)
history = pipeline.server.run()

results = pipeline.auto_test()   # per-node metrics, stored in the database
```

Results can be visualized with `MEDfl.LearningManager.plot.AccuracyLossPlotter`, and hyperparameters tuned with `MEDfl.LearningManager.params_optimiser.ParamsOptimiser` (grid search or Optuna).

Step-by-step notebooks are available in [notebooks/tutorial/](notebooks/tutorial/) (database, NetManager, transfer learning, LearningManager, results, model choice, hyperparameter optimization).

---

## Package Structure

```
MEDfl/
├── NetManager/            # Simulation: networks, nodes, datasets, FL setups, DB access
├── LearningManager/       # Simulation: models, strategies, server/client, pipelines, plots, HPO
│   ├── xgboost/           #   federated XGBoost (simulation)
│   └── shap/              #   SHAP config, local explainers, federated aggregation
├── rw/                    # Real-world deployment
│   ├── server.py          #   FederatedServer
│   ├── client.py          #   FlowerClient + DPConfig (PyTorch)
│   ├── strategy.py        #   Strategy wrapper (config push, logging, checkpointing)
│   ├── model.py           #   Configurable MLP (Net)
│   ├── holdout.py         #   Federated holdout test & threshold optimization
│   └── xgboost/           #   XGBoostStrategy, XGBoostFlowerClient, XGBoostConfig
└── scripts/               # Database creation utilities
```

---

## Documentation

Full documentation, including API reference and complete simulation and real-world tutorials, is built with **Sphinx** from [docs/](docs/). To build and browse it locally:

```bash
cd docs
make clean && make html
python -m http.server --directory _build/html
```

---

## Testing

```bash
pip install pytest
pytest tests/
```

---

## Contributing

Contributions of all kinds are welcome — bug reports, new features, documentation and tutorials.

1. Fork the repository and create a feature branch from `main`.
2. Add tests for your changes and make sure `pytest` passes.
3. Format your code with `black` and check it with `flake8`.
4. Open a Pull Request describing the motivation and the changes.

Please use [GitHub Issues](https://github.com/MEDomics-UdeS/MEDfl/issues) to report bugs or request features.

---

## Acknowledgment

MEDfl is part of the **MEDomicsLab** initiative at the **Université de Sherbrooke**. It was developed to enable **secure, privacy-preserving and reproducible** machine learning across distributed medical datasets.

---

## Authors

- [Ouael Nedjem Eddine Sahbi](https://github.com/ouaelesi) — Master's Research Student, Université de Sherbrooke
- [Haithem Lamri](https://github.com/HaithemLamri) — Research Intern
- [MEDomics-UdeS](https://www.medomics-udes.org/en/)

---

## License

MEDfl is distributed under the **GNU General Public License v3.0**. See [LICENSE](LICENSE) for details.

<div align="center">

⭐ If MEDfl is useful to your work, please consider starring the repository to support its development.

</div>
