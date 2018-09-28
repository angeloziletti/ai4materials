.. highlight:: shell

============
Installation
============


From sources
------------

The only `ai4materials` dependence that needs to be manually installed is `condor`_ (a package to calculate diffraction
intensities).
You can find details on how to install condor `here <http://lmb.icm.uu.se/condor/static/docs/installation.html#dependencies>`_.
Make sure you install the `condor` dependencies `libspimage` and `spsim` as well.
Once you have successfully installed condor, you can proceed with the `ai4materials` installation.

The sources for ai4materials can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/angeloziletti/ai4materials

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/angeloziletti/ai4materials/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _condor: http://lmb.icm.uu.se/condor/static/docs/index.html
.. _Github repo: https://github.com/angeloziletti/ai4materials
.. _tarball: https://github.com/angeloziletti/ai4materials/tarball/master



.. sectionauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>



