Installation
============

Python installation
-------------------
The MEDfl package requires ``python 3.9`` or more to be run. If you don't have it installed  on your machine, check out the following link  `Python <https://www.python.org/downloads/>`_
It also requires MySQL database

Package Installation
--------------------

For now, you can install the ``MEDfl`` package as follows:

.. code-block:: bash

   git clone https://github.com/MEDomics-UdeS/MEDfl.git
   cd MEDfl
   pip install -e .


MySQL DB Configuration
----------------------

MEDfl requires a MySQL DB connection, and this is in order to allow users to work with their own tabular datasets. We have created a bash script to install and configure a MySQL DB with phpMyAdmin monitoring system. Run the following command, then change your credentials in the MEDfl/scripts/base.py file:

.. code-block:: bash

   sudo bash MEDfl/scripts/setup_mysql.sh
