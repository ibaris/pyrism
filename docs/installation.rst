Installation
============
There are currently different methods to install `prism`.

Using pip
---------
The ` prism ` package is provided on pip. You can install it with::

    pip install prism

Standard Python
---------------
You can also download the source code package from this repository or from pip. Unpack the file you obtained into some directory (it can be a temporary directory) and then run::

    python setup.py install
  
Test installation success
-------------------------
Independent how you installed ` prism `, you should test that it was sucessfull by the following tests::

    python -c "from prism import I2EM"

If you don't get an error message, the module import was sucessfull.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
