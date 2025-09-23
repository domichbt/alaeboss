"""Code to obtain imaging systematics weights for the galaxies of a spectroscopic survey catalog. The code implements linear regression on the histograms of the systematics maps, reproducing the "linear" method of recent DESI and eBOSS catalogs."""

from .linear_regression import LinearRegressor
