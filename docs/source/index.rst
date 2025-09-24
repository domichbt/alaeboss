.. alaeboss documentation master file, created by
   sphinx-quickstart on Mon Sep 22 15:21:27 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

alaeboss documentation
======================

Use "à la eBOSS" linear regression on data and randoms catalogs to correct for imaging systematics via weights to apply to the galaxies.

The code is designed to mimic the original eBOSS method but is implemented more efficiently. In particular, it makes use of JAX to accelerate the minimization process, and pre-made scripts such as :py:func:`~alaeboss.produce_imweights` take care to minimize the amount of I/O.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

