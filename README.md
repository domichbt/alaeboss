Code to obtain imaging systematics weights for the galaxies of a spectroscopic survey catalog. The code implements linear regression on the histograms of the systematics maps, reproducing the "linear" method of recent DESI and eBOSS catalogs.

Module `alaeboss.py` implements the regression, providing a huge speed-up compared to the former implementation by using JAX and feeding the gradient to Minuit.
The `produce_imweights.py` script gives an example wrapper function that runs the regression on DESI data in an efficient way (minimal amount of I/O and random catalog manipulation).
