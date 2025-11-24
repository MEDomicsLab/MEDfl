Documentation
=============

We used Sphinx to create the documentation for this project. You can generate and host it locally by compiling the documentation source code using:

.. code-block:: bash

   cd docs
   make clean
   make html

Then open it locally using:

.. code-block:: bash

   cd _build/html
   python -m http.server