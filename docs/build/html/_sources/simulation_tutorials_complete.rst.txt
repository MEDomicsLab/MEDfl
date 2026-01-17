MEDfl Complete Tutorial (Simulation)
====================================

In this complete tutorial, we will demonstrate how to use the ``MEDfl`` package
to set up and run a **federated learning experiment in simulation mode**.

Starting from a *realistic healthcare scenario*, we will:

* Configure the **database** used by MEDfl
* Create a **network** and **nodes** with the ``NetManager``
* Generate a **federated dataset**
* Define a **dynamic model**
* Configure the **aggregation strategy**
* Start a **Flower-based FL server**
* Run the **federated training pipeline**
* Plot **accuracy and loss**
* Automatically **test** the final model and store results in the database

This tutorial is based on the accompanying Jupyter notebook. It is designed as a
step-by-step guide you can follow and adapt to your own datasets and
configurations.


Real-world motivation
---------------------

**Martin** is an AI researcher whose main interest is applying AI to the healthcare
domain. He is contacted by a prestigious institute to study the feasibility of a
new project:

    Designing and developing a federated learning system between several hospitals,
    using deep learning while preserving patient privacy.

After analyzing the requirements, Martin identifies that the project needs:

* **Federated Learning (FL)** to keep data local to each hospital
* **Differential Privacy (DP)** to protect model updates
* A robust **data and experiment management** layer

Martin knows ``MEDfl`` has been designed for exactly these kinds of tasks.  
With its two main sub-packages, ``NetManager`` and ``LearningManager``, MEDfl
allows him to:

* Design different **federated learning architectures (setups)**
* Simulate **real-world collaborations** between hospitals
* Integrate **transfer learning** and **differential privacy**
* Store and compare **results** systematically in a database


0. Prerequisites
----------------

Before following this tutorial, make sure you have:

* Installed ``MEDfl`` and its dependencies (see :doc:`installation`)
* A Python environment (e.g. ``fl-env``) with:

  * ``torch``
  * ``flwr``
  * ``pandas``
  * ``sqlalchemy``

* A CSV dataset. In this tutorial we use a diabetes dataset located at::

    ../data/masterDataSet/diabetes_dataset.csv

.. note::

   In production, MEDfl can be connected to a **MySQL** database
   (see :doc:`database_management`).  
   In this tutorial, for simplicity, we use a **local SQLite database**.


1. Environment and imports
--------------------------

We start by making sure the project root is on the Python path and importing all
the necessary modules.

.. code-block:: python

   import sys
   sys.path.append("../..")

   import os
   os.environ["PYTHONPATH"] = "../.."

   # Database and data handling
   import pandas as pd

   # Torch imports
   import torch
   import torch.nn as nn
   import torch.optim as optim

   # Flower
   import flwr as fl

   # MEDfl imports - NetManager
   from MEDfl.NetManager.node import Node
   from MEDfl.NetManager.network import Network
   from MEDfl.NetManager.flsetup import FLsetup
   from MEDfl.NetManager.database_connector import DatabaseManager

   # MEDfl imports - LearningManager
   from MEDfl.LearningManager.dynamicModal import DynamicModel
   from MEDfl.LearningManager.model import Model
   from MEDfl.LearningManager.strategy import Strategy
   from MEDfl.LearningManager.server import FlowerServer
   from MEDfl.LearningManager.flpipeline import FLpipeline
   from MEDfl.LearningManager.plot import AccuracyLossPlotter
   from MEDfl.LearningManager.utils import set_db_config


2. Database configuration
-------------------------

In MEDfl, all networks, nodes, datasets, setups, pipelines, and results are
stored in a relational database.

In this tutorial we use a local SQLite database file named
``medfl_database.db``:

.. code-block:: python

   # Configure the database path
   set_db_config("./medfl_database.db")

   # Create and connect the database manager
   db_manager = DatabaseManager()
   db_manager.connect()
   connection = db_manager.get_connection()

   print("Database connection OK")

Next, we generate the necessary MEDfl tables based on a **master dataset** CSV
file. This file describes the global structure of the data that will later be
partitioned across hospitals.

.. code-block:: python

   db_manager.create_MEDfl_db(
       path_to_csv="../data/masterDataSet/diabetes_dataset.csv"
   )

.. note::

   ``create_MEDfl_db``:

   * infers dataset-related tables from the CSV structure,
   * creates the core MEDfl tables to manage networks, nodes, datasets and
     experiments.


3. Network creation (NetManager)
--------------------------------

We now create a **federated network** that will hold all hospitals (nodes) and
the corresponding datasets.

.. code-block:: python

   # Create a new network
   net = Network("Net1")

   # Register the network in the database
   net.create_network()

   print(net.name)  # "Net1"

We then register the **master dataset** associated with this network:

.. code-block:: python

   net.create_master_dataset(
       "../data/masterDataSet/diabetes_dataset.csv"
   )


4. Federated Learning setup (FLsetup)
-------------------------------------

An ``FLsetup`` describes a **federated learning configuration** for a given
network: which network it uses, how datasets are split, and how the federated
dataset will be derived.

Here we create an automatic setup:

.. code-block:: python

   auto_fl = FLsetup(
       name="Flsetup_2",
       description="The second FL setup",
       network=net,
   )
   auto_fl.create()

   auto_fl.list_allsetups()

This will show a table of FL setups stored in the database, including the one we
just created.


5. Node creation and dataset upload
-----------------------------------

Now we add **hospital nodes** to the network. Each node receives a local
dataset, representing that hospital’s data.

.. code-block:: python

   # Train node: hospital_1
   hospital_1 = Node(name="hospital_1", train=1)
   net.add_node(hospital_1)
   hospital_1.upload_dataset(
       "hospital_1",
       "../data/masterDataSet/client_1_dataset.csv",
   )

.. code-block:: python

   # Train node: hospital_2
   hospital_2 = Node(name="hospital_2", train=1)
   net.add_node(hospital_2)
   hospital_2.upload_dataset(
       "hospital_2",
       "../data/masterDataSet/client_2_dataset.csv",
   )

.. code-block:: python

   # Test node: hospital_3 (no local training)
   hospital_3 = Node(name="hospital_3", train=0)
   net.add_node(hospital_3)
   hospital_3.upload_dataset(
       "hospital_3",
       "../data/masterDataSet/client_3_dataset.csv",
   )

You can list all nodes registered in the network:

.. code-block:: python

   net.list_allnodes()


6. Federated dataset creation
-----------------------------

We now ask MEDfl to build a **federated dataset** from:

* the FL setup,
* the nodes,
* and the master dataset.

In this example, we consider ``"diabetes"`` as the target variable.

.. code-block:: python

   fl_dataset = auto_fl.create_federated_dataset(
       output="diabetes",   # target column
       fit_encode=[],       # columns to encode (if any)
       to_drop=["diabetes"] # columns to drop from the inputs
   )

You can inspect the federated dataset object:

.. code-block:: python

   fl_dataset.size          # number of clients / partitions

.. code-block:: python

   auto_fl.get_flDataSet()  # summary table stored in the DB


7. Model definition (DynamicModel)
----------------------------------

MEDfl provides a ``DynamicModel`` class to create models dynamically depending
on the task (binary classification, multiclass, regression, etc.).

In this tutorial, we build a **binary classifier** with 8 input features:

.. code-block:: python

   # Create a DynamicModel helper
   dynamic_model = DynamicModel()

   # Build a specific model
   specific_model = dynamic_model.create_model(
       model_type="Binary Classifier",
       params_dict={
           "input_dim": 8,
           "output_dim": 1,
           "hidden_dims": [16, 32],
       },
   )

   # Optimizer and loss
   optimizer = optim.SGD(specific_model.parameters(), lr=0.001)
   criterion = nn.BCELoss()

   # Wrap everything into a MEDfl Model
   global_model = Model(specific_model, optimizer, criterion)

   # Initial parameters (to share with clients)
   init_params = global_model.get_parameters()


8. Aggregation strategy
-----------------------

The aggregation strategy specifies how local model updates are combined on the
server side (e.g., FedAvg, FedAdam, etc.).

Here we use ``FedAdam`` as an example:

.. code-block:: python

   aggreg_algo = Strategy(
       "FedAdam",
       fraction_fit=1.0,
       fraction_evaluate=1.0,
       min_fit_clients=2,
       min_evaluate_clients=2,
       min_available_clients=2,
       initial_parameters=init_params,
   )
   aggreg_algo.create_strategy()


9. Federated learning server
----------------------------

We now create the **Flower-based federated server** that will orchestrate
training across the clients (nodes) using the federated dataset.

.. code-block:: python

   server = FlowerServer(
       global_model,
       strategy=aggreg_algo,
       num_rounds=10,
       num_clients=len(fl_dataset.trainloaders),
       fed_dataset=fl_dataset,
       diff_privacy=False,  # set True to enable DP
       client_resources={
           "num_cpus": 1.0,
           "num_gpus": 0.0,
       },
   )


10. FL pipeline creation and training
-------------------------------------

To make the experiment reproducible and easy to manage, MEDfl provides the
``FLpipeline`` class. It links the server, setup, and results together.

.. code-block:: python

   ppl_1 = FLpipeline(
       name="the first fl_pipeline",
       description="This is our first FL pipeline",
       server=server,
   )

To start federated training:

.. code-block:: python

   history = ppl_1.server.run()


11. Plotting accuracy and loss
------------------------------

After training, we can visualize the evolution of global accuracy and loss
across federated rounds.

.. code-block:: python

   global_accuracy = ppl_1.server.accuracies
   global_loss = ppl_1.server.losses

   results_dict = {
       ("LR: 0.001, Optimizer: SGD", "accuracy"): global_accuracy,
       ("LR: 0.001, Optimizer: SGD", "loss"): global_loss,
   }

   plotter = AccuracyLossPlotter(results_dict)
   plotter.plot_accuracy_loss()

This produces a figure showing the training curves over the rounds, helping you
compare different configurations or hyperparameters.


12. Automatic testing and result storage
----------------------------------------

Finally, we can automatically test the global model on **test nodes** and store
the metrics in the database:

.. code-block:: python

   test_results = ppl_1.auto_test()
   test_results

Each entry in ``test_results`` contains:

* The **node name**
* A **classification report** including:

  * Confusion matrix (TP, FP, FN, TN)
  * Accuracy
  * Sensitivity/Recall
  * Specificity
  * PPV/Precision
  * NPV
  * F1-score
  * False positive rate
  * True positive rate
  * AUC

All these results are also saved in the MEDfl database, allowing you to:

* Compare different FL setups
* Track experiments across time
* Reuse configurations in future studies




