
Connecting Clients Behind Private Networks Using Tailscale
===========================================================

In real-world federated learning experiments, client machines (e.g. hospitals,
institutions, or secured servers) are often located behind **private networks**:

* NAT-ed home or institutional routers
* Private subnets without public IPs
* Firewalls blocking inbound connections
* VPNs that isolate machines from the public Internet

In such situations, the **central federated server cannot directly reach the
clients**, even if the server itself has a public IP address.

To solve this issue, MEDfl supports running experiments over a **private overlay
network** using **Tailscale**.

Why Tailscale?
--------------

Tailscale creates a **secure, peer-to-peer virtual private network (VPN)** between
machines, based on WireGuard. It allows machines to communicate as if they were
on the same local network, even if they are:

* behind NATs
* on different physical networks
* distributed across institutions or countries

Key advantages:

* No port forwarding required
* No firewall reconfiguration in most cases
* End-to-end encryption by default
* Each machine receives a **stable private IP address**

.. image:: figures/Images/tailscale_architecture_placeholder.png
   :width: 70%
   :align: center
   :alt: Tailscale overlay network architecture (placeholder)

   *(Placeholder: diagram showing server and clients connected via Tailscale)*


Prerequisites
-------------

Before starting, ensure that:

* You have access to all machines (server + clients)
* Python and MEDfl are already installed on each machine
* You can install third-party software (Tailscale)

You will need:

* A **Tailscale account**
* Internet access on all machines (outbound is sufficient)

Creating a Tailscale Account
----------------------------

1. Go to the official Tailscale website:

   .. code-block:: text

      https://tailscale.com

2. Create an account using one of the supported identity providers
   (Google, GitHub, Microsoft, etc.).

.. image:: figures/Images/tailscale_signup_placeholder.png
   :width: 60%
   :align: center
   :alt: Tailscale signup page (placeholder)

   *(Placeholder: screenshot of Tailscale signup page)*


Installing Tailscale on Each Machine
------------------------------------

Tailscale must be installed on **all machines** involved in the experiment:

* The **central server**
* Every **federated client**

### On Linux

.. code-block:: bash

   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up

### On Windows

1. Download the installer from:

   .. code-block:: text

      https://tailscale.com/download

2. Run the installer and follow the setup wizard.
3. Launch Tailscale and log in.

.. image:: figures/Images/tailscale_windows_install_placeholder.png
   :width: 60%
   :align: center
   :alt: Tailscale Windows installer (placeholder)

   *(Placeholder: screenshot of Tailscale Windows installer)*

### On macOS

.. code-block:: bash

   brew install --cask tailscale
   sudo tailscale up


Adding Machines to the Tailnet
------------------------------

After logging in on each machine:

* All machines will automatically join the same **tailnet**
  (Tailscale private network associated with your account).
* Each machine is assigned a **unique private IP address** in the form:

  .. code-block:: text

     100.x.y.z

You can list connected machines using:

.. code-block:: bash

   tailscale status

Example output:

.. code-block:: text

   100.65.215.27   server-node        linux   active
   100.72.88.14    client-hospital-1  windows active
   100.91.33.52    client-hospital-2  linux   active

.. image:: figures/Images/tailscale_admin_console_placeholder.png
   :width: 80%
   :align: center
   :alt: Tailscale admin console (placeholder)

   *(Placeholder: screenshot of Tailscale admin console showing machines)*


Using Tailscale IP Addresses in MEDfl
-------------------------------------

Once all machines are connected via Tailscale, **you must use the Tailscale IP
addresses instead of local or public IPs** when configuring MEDfl.

### Server configuration

On the server machine, the server listens as usual:

.. code-block:: python

   server = FederatedServer(
       host="0.0.0.0",
       port=8080,
       num_rounds=10,
       strategy=custom_strategy,
   )

The server does **not** need to know it is running behind Tailscale.

### Client configuration (important)

On each client machine, update the ``server_address`` to use the
**Tailscale IP of the server**:

.. code-block:: python

   client = FlowerClient(
       server_address="100.65.215.27:8080",  # Tailscale IP of the server
       data_path="../data/client1_with_id.csv",
       seed=42,
       dp_config=None,
   )

This ensures that:

* Clients can reach the server even behind private networks
* No public IP or port forwarding is required

.. image:: figures/Images/tailscale_ip_usage_placeholder.png
   :width: 70%
   :align: center
   :alt: Using Tailscale IPs in configuration (placeholder)

   *(Placeholder: diagram showing clients connecting to server via Tailscale IP)*


Security Considerations
-----------------------

* All communication over Tailscale is **encrypted by default**
* Only machines authorized in your tailnet can connect
* You can further restrict access using:
  * Tailscale ACLs
  * Device tags
  * Ephemeral nodes (for temporary clients)

This makes Tailscale well-suited for **medical and sensitive data environments**.


Summary
-------

When clients are located behind private VPNs or unreachable networks:

* Direct connections to the server may fail
* Tailscale provides a simple and secure solution
* Steps are:

  1. Create a Tailscale account
  2. Install Tailscale on server and clients
  3. Add all machines to the same tailnet
  4. Use **Tailscale IP addresses** in MEDfl configuration

This approach enables **true real-world federated learning experiments**
across institutions without modifying existing network infrastructures.


Real-world Federated Learning Tutorial
======================================

This tutorial shows how to use the ``MEDfl.rw`` (real-world) module to run a
**true multi-machine federated learning experiment**:

* One **central server** (orchestrator)
* Several **remote clients** (hospitals / institutions)
* Each client keeps its data **local** in a CSV file
* The server enforces a **shared schema** (same features/target on all sites)
* Validation / test splits are done **per client**, with optional overrides
* Optional **Differential Privacy (DP)** configuration per client


High-level architecture
-----------------------

In real-world mode, MEDfl is organized as follows:

* :mod:`MEDfl.rw.server`
  
  * ``FederatedServer``: wraps the Flower server and coordinates the training
  * ``Strategy``: defines the aggregation and training hyperparameters

* :mod:`MEDfl.rw.client`
  
  * ``FlowerClient``: local client that loads CSV data, applies splits, trains
  * ``DPConfig``: optional configuration for differential privacy

.. image:: figures/Images/rw_files.png
   :width: 30%
   :align: center

The typical workflow is:

1. Prepare a **CSV dataset** on each client machine.
2. Configure and start the **server** on a reachable host.
3. Configure and start the **clients**, pointing them to the server.
4. Monitor federated rounds and collect saved models and metrics.


1. Preparing client data
------------------------

Each client will load data from a **local CSV file**. All clients must share:

* The same **feature columns**
* The same **target column**
* An optional **ID column** (for split_mode based on IDs)

Example structure of a client CSV file:

.. code-block:: text

   id,MajorAxisLength,Area,Eccentricity,ConvexArea,label
   0,134.12,10023,0.88,10450,1
   1,120.57,8900,0.76,9100,0
   2,143.34,11010,0.91,11300,1
   ...

In this tutorial, on the client machine we will use:

.. code-block:: text

   ../data/client1_with_id.csv

with:

* ``id`` as the identifier
* ``MajorAxisLength, Area, Eccentricity, ConvexArea`` as features
* ``label`` as the target column


2. Server configuration
-----------------------

Create a file called :file:`run_server.py` on the **server machine**.

.. code-block:: python

   # run_server.py
   from MEDfl.rw.server import FederatedServer, Strategy

   # Optional: build a list of test IDs (here 0..1499)
   test_ids = list(range(1500))

   # Optional per-client overrides:
   # keys must match the hostname seen by MEDfl on each client machine.
   client_fractions = {
       "dinf-medomi-05b": {
           "val_fraction": 0.10,
           "test_ids": list(range(1500)),  # here: all IDs used for test
       }
   }

   custom_strategy = Strategy(
       name="FedAvg",
       fraction_fit=1.0,
       min_fit_clients=1,
       min_evaluate_clients=1,
       min_available_clients=1,

       local_epochs=10,
       threshold=0.5,
       learning_rate=0.01,
       optimizer_name="SGD",

       saveOnRounds=3,        # save the model every 3 rounds
       savingPath="./",       # where to store the checkpoints
       total_rounds=10,

       # --- Data schema enforcement ---
       features="MajorAxisLength,Area,Eccentricity,ConvexArea",  # comma-separated feature names
       target="label",                                           # target column name

       # --- Default split configuration (can be overridden per client) ---
       val_fraction=0.15,
       test_fraction=0.10,

       # --- Splitting strategy ---
       split_mode="per_client",  # split done independently on each client
       id_col="id",              # ID column used for test_ids-based splits

       # --- Optional per-client overrides ---
       client_fractions=client_fractions,
   )

   server = FederatedServer(
       host="0.0.0.0",   # listen on all interfaces
       port=8080,        # port to expose to clients
       num_rounds=10,
       strategy=custom_strategy,
   )

   if __name__ == "__main__":
       server.start()

Key parameters
~~~~~~~~~~~~~~

* ``features``: comma-separated list of feature names expected in each client CSV.
* ``target``: target column name for supervised learning.
* ``split_mode="per_client"``: each client performs its own train/val/test split.
* ``id_col``: column name containing sample IDs. Used when you want to define
  test sets using ``test_ids``.
* ``val_fraction`` and ``test_fraction``: default fractions of data used for
  validation and testing on each client.
* ``client_fractions``: per-client overrides, indexed by the **hostname** seen
  by MEDfl on that machine. You can override:

  * ``val_fraction``
  * ``test_fraction``
  * ``test_ids`` (list of IDs that must go to the test set)

* ``saveOnRounds`` and ``savingPath``: control how often and where the model
  checkpoints are saved.


3. Client configuration
-----------------------

On each **client machine** (e.g. each hospital), create a file
:file:`run_client.py` (or :file:`run_client_override_splits.py`) similar to:

.. code-block:: python

   # run_client_override_splits.py
   from MEDfl.rw.client import FlowerClient, DPConfig

   client = FlowerClient(
       server_address="100.65.215.27:8080",  # IP:port of the server
       data_path="../data/client1_with_id.csv",
       seed=42,
       dp_config=None,                      # or a DPConfig instance (see below)
   )

   if __name__ == "__main__":
       client.start()

Parameters:

* ``server_address``: must point to the **public / reachable address** of the
  server (as configured in ``FederatedServer``).
* ``data_path``: local path to the CSV file on this client machine.
* ``seed``: used for deterministic splits.
* ``dp_config``: optional differential privacy configuration (see next subsection).


4. Enabling Differential Privacy (optional)
-------------------------------------------

To enable differential privacy on a client, create a :class:`DPConfig` instance
and pass it to ``FlowerClient``:

.. code-block:: python

   from MEDfl.rw.client import FlowerClient, DPConfig

   dp_conf = DPConfig(
       noise_multiplier=1.0,
       max_grad_norm=1.0,
       batch_size=32,
       secure_rng=False,
   )

   client = FlowerClient(
       server_address="100.65.215.27:8080",
       data_path="../data/client1_with_id.csv",
       seed=42,
       dp_config=dp_conf,
   )

   if __name__ == "__main__":
       client.start()

Typical hyperparameters:

* ``noise_multiplier``: controls the magnitude of the added noise (higher =
  more privacy, more utility loss).
* ``max_grad_norm``: gradient clipping norm.
* ``batch_size``: local training batch size.
* ``secure_rng``: whether to use a cryptographically secure RNG.


5. Running the experiment
-------------------------

Once both sides are configured:

1. **On the server machine**:

   .. code-block:: bash

      python run_server.py

   The server will start and wait for clients to connect.

2. **On each client machine**:

   .. code-block:: bash

      python run_client_override_splits.py

   Each client will:

   * Load its local CSV data
   * Apply the default or overridden splits
   * Participate in federated training rounds coordinated by the server

3. During training:

   * The server runs for ``num_rounds`` (here 10).
   * Every ``saveOnRounds`` rounds (here 3), a checkpoint is written to
     ``savingPath`` on the server (`./` in the example).
   * Metrics (loss, accuracy, etc.) are logged via the strategy / server
     logic implemented in :mod:`MEDfl.rw`.


6. Customizing splits per client
--------------------------------

Sometimes you want **fine-grained control** over how each client splits its
data. MEDfl allows this through the ``client_fractions`` dictionary.

Example: a client with hostname ``dinf-medomi-05b`` should:

* Use **10%** of its data as validation,
* Use a specific set of IDs as test data.

.. code-block:: python

   client_fractions = {
       "dinf-medomi-05b": {
           "val_fraction": 0.10,
           "test_ids": list(range(1500)),  # IDs 0..1499
       }
   }

   custom_strategy = Strategy(
       # ...
       client_fractions=client_fractions,
   )

On that specific machine:

* The effective validation fraction will be 0.10 (instead of the default 0.15).
* Samples whose ``id`` is in ``test_ids`` will be forced into the test set.

.. note::

   Make sure that:

   * The hostname key (e.g. ``"dinf-medomi-05b"``) exactly matches the name
     seen by MEDfl on the client machine.
   * The ``id_col`` in the strategy configuration matches the ID column in the
     client CSV files.


7. Summary
----------

In this real-world tutorial, you have learned how to:

* Prepare local CSV datasets for each client
* Configure a **FederatedServer** with:

  * a specific aggregation strategy (``FedAvg``)
  * enforced feature/target schema
  * default and per-client validation/test splits
  * checkpoint saving

* Configure and run **FlowerClient** instances on remote machines
* Optionally enable **Differential Privacy** using ``DPConfig``

This setup corresponds to a **true multi-institution environment** where data
never leaves the client machines, and only model updates are exchanged.



