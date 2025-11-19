Configuration File
==================

Project base url
----------------
In the file ``MEDfl/global_params.yaml``, you need to update the base URL parameters according to the local path of your project.

.. code-block:: yaml

   base_url: 'PATH_TO_PROJECT/MEDfl'


MySQL Configuration
-------------------
In the file ``MEDfl/scripts/db_config.ini``, you can specify the SQL connection parameters.

.. code-block:: bash

   [mysql]
    host = localhost
    port = 3306
    user = your_username
    password = your_password
    database = MEDfl

Also, in the file ``MEDfl/scripts/create_db.py``:

.. code-block:: python

    mydb = mysql.connector.connect(host="localhost", user="your_username", password="your_password")

Learning Parameters
-------------------
In the file ``MEDfl/MEDfl/LearningManager/params.yaml``, you can modify the parameters for creating your model.

.. code-block:: yaml

    task: BinaryClassification
    optimizer: SGD
    train_batch_size: 32
    test_batch_size: 1
    train_epochs: 6
    lr: 0.001
    diff_privacy: True
    MAX_GRAD_NORM: 1.0
    EPSILON: 20.0
    DELTA: 1e-5
    num_rounds: 3
    min_evalclient: 2

DataSets
--------
In the file ``/MEDfl/MEDfl/LearningManager/params.yaml``, you can specify the path to the CSV files of the dataset you want to use.

.. code-block:: yaml

    path_to_master_csv: 'PATH TO YOUR MASTER CSV'
    path_to_test_csv: 'PATH TO YOUR TEST CSV'

.. note::
    The ``path_to_master_csv`` is the CSV file used to create the **MasterDataSet**
