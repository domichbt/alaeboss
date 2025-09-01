"""
Utility function to regress imaging systematic weights on DESI data as is done in `mkCat_main.py` using methods in alaeboss.py.

Correspondence between arguments and variables in `mkCat_main.py`:

* `templates_maps_path_N = f"{lssmapdirout}{tpstr}_mapprops_healpix_nested_nside{nside}_N.fits"`
* `templates_maps_path_S = f"{lssmapdirout}{tpstr}_mapprops_healpix_nested_nside{nside}_S.fits"`
* `redshift_range = zrl`
* `tracer_type = type`
* `output_directory = dirout`
* `fit_maps`: the maps you want to regress against. Might be usemaps if not None, otherwise `mainp.fit_maps_allebv` (LRGs) or `mainp.fit_maps`
* `output_directory = dirout`

You can setup a basic logger to pass to the function as
```
logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
```
"""

import logging
from pathlib import Path
from time import time

import fitsio
import jax
import LSS.common_tools as common
import numpy as np
from astropy.table import Table
from LSS.imaging import densvar

from alaeboss import LinearRegressor


def produce_imweights(
    # Input and output control
    data_catalog_path: str,
    random_catalogs_paths: list[str],
    tracer_type: str,
    redshift_range: list[(float, float)],
    templates_maps_path_S: str,
    templates_maps_path_N: str,
    fit_maps: list[str],
    output_directory: str,
    output_column_name: str = "WEIGHT_IMLIN",
    save_summary_plots: bool = True,
    # Regression-specific arguments
    nbins: int = 10,
    tail: float = 0.5,
    # Miscellaneous
    logger: logging.Logger = None,
    loglevel: str = "INFO",
):
    """
    Perform linear regression to compute imaging systematics weights for a given tracer type, data catalog, random catalogs, set of maps.
    This function reads in a data catalog and associated random catalogs, applies selection criteria, loads imaging systematics templates, and performs regression to estimate and assign imaging weights to the data. The regression is performed separately for different photometric regions and redshift bins. Optionally, summary plots can be saved.

    Parameters
    ----------
    data_catalog_path : str
        Path to the input data catalog FITS file.
    random_catalogs_paths : list[str]
        List of paths to random catalogs FITS files.
    tracer_type : str
        Type of tracer (e.g., 'LRG', 'ELG_LOP', 'QSO').
    redshift_range : list of tuple of float
        List of (z_min, z_max) tuples defining redshift bins for regression.
    templates_maps_path_S : str
        Path to the South region imaging systematics templates file.
    templates_maps_path_N : str
        Path to the North region imaging systematics templates file.
    fit_maps : list[str]
        List of template map names to use in the regression.
    output_directory : str
        Directory where output plots and parameter files will be saved.
    output_column_name : str, optional
        Name of the output column to store computed weights in the data catalog (default is "WEIGHT_IMLIN").
    save_summary_plots : bool, optional
        Whether to save summary plots of the regression results (default is True).
    nbins : int, optional
        Number of bins to use in regression preparation (default is 10).
    tail : float, optional
        Fraction of data to cut as outliers from each tail (default is 0.5).
    logger : logging.Logger, optional
        Logger object for logging progress and information (default is None). This will not log anything if set to None.
    loglevel : str, optional
        Logging level for the regressor's logger (default is "INFO"). Will not affect `logger`'s level.
    Returns
    -------
    None
        The function modifies the input data catalog in place by adding or updating the output_column_name
        with computed imaging weights, and writes regression parameters and plots to the output directory.
    Notes
    -----
    Loading some columns only from FITS file during NERSC jobs can be very long for mysterious reasons. If you are experiencing huge catalog readtimes, this might be why.

    """
    logger = logger or logging.getLogger("dummy")

    logger.info("Doing linear regression for imaging systematics")

    jax.config.update("jax_enable_x64", True)
    logger.info("Enabled 64-bit mode for JAX")

    time_start = time()

    debv = common.get_debv()  # for later
    sky_g, sky_r, sky_z = common.get_skyres()
    output_directory = Path(output_directory)

    # read data catalogs
    logger.info("Reading data catalogs")
    all_data = Table(fitsio.read(data_catalog_path))
    # read randoms catalogs (note that since we are reading a subset of columns, this can take a lot on time from a job, no idea why)
    logger.info("Reading %i randoms catalogs", len(random_catalogs_paths))
    rands = np.concatenate(
        [
            fitsio.read(
                random_catalog_path,
                columns=["RA", "DEC", "PHOTSYS"],
            )
            for random_catalog_path in random_catalogs_paths
        ]
    )

    # select good data that has been observed
    logger.info("Selecting good and observed data")
    data_selection = common.goodz_infull(tracer_type[:3], all_data) & (
        all_data["ZWARN"] != 999999
    )
    dat = all_data[data_selection]

    # prepare array to receive computed weights
    weights_imlin = np.ones(len(dat), dtype=float)

    # define photometric regions
    photometric_regions = ["S", "N"]
    if tracer_type == "QSO":
        photometric_regions = ["DES", "SnotDES", "N"]

    for region in photometric_regions:
        # Are we north or south? (irrespective of DES)
        northsouth = region
        if region in ["DES", "SnotDES"]:
            northsouth = "S"

        # get healpix maps for the systematics
        logger.info("Loading healpix templates for region %s", northsouth)
        match northsouth:
            case "S":
                sys_tab = Table.read(templates_maps_path_S)
            case "N":
                sys_tab = Table.read(templates_maps_path_N)
            case _:
                raise KeyError(
                    "Value %s is not valid as a template region (North or South, ie 'S' or 'N')",
                    northsouth,
                )

        cols = list(sys_tab.dtype.names)  # names of templates

        for col in cols:
            if "DEPTH" in col:
                bnd = col.split("_")[-1]
                sys_tab[col] *= 10 ** (-0.4 * common.ext_coeff[bnd] * sys_tab["EBV"])
        for ec in ["GR", "RZ"]:
            sys_tab["EBV_DIFF_" + ec] = debv["EBV_DIFF_" + ec]
        if "EBV_DIFF_MPF" in fit_maps:
            sys_tab["EBV_DIFF_MPF"] = sys_tab["EBV"] - sys_tab["EBV_MPF_Mean_FW15"]
        if "SKY_RES_G" in fit_maps:
            sys_tab["SKY_RES_G"] = sky_g[northsouth]
        if "SKY_RES_R" in fit_maps:
            sys_tab["SKY_RES_R"] = sky_r[northsouth]
        if "SKY_RES_Z" in fit_maps:
            sys_tab["SKY_RES_Z"] = sky_z[northsouth]

        logger.info(f"Maps for regression: {fit_maps}")

        # select randoms now and retrieve systematics
        logger.info("Masking randoms according to the region")
        match region:
            case "N" | "S":
                region_mask_randoms = rands["PHOTSYS"] == region
                region_mask_data = dat["PHOTSYS"] == region
            case "DES":
                region_mask_randoms = common.select_regressis_DES(rands)
                region_mask_data = common.select_regressis_DES(dat)
            case "SnotDES":
                region_mask_randoms = (rands["PHOTSYS"] == "S") & (
                    ~common.select_regressis_DES(rands)
                )
                region_mask_data = (dat["PHOTSYS"] == "S") & (
                    ~common.select_regressis_DES(dat)
                )
            case _:
                logger.info("other regions not currently supported")
                raise NotImplementedError("Exiting due to critical error with region")
        region_randoms = rands[region_mask_randoms]
        # rand_syst = densvar.read_systematic_maps_alt(region_randoms['RA'], region_randoms['DEC'], sys_tab, use_maps)
        logger.info("Reading template values for the randoms")
        randoms_templates_values = densvar.read_systematic_templates_stacked_alt(
            region_randoms["RA"],
            region_randoms["DEC"],
            sys_tab,
            fit_maps,
        )
        logger.info(
            f"Preparation for region {region} is done. Starting regressions per redshift slide."
        )

        for z_range in redshift_range:
            logger.info(
                f"Getting weights for region {region} and redshift bin {z_range[0]} < z < {z_range[1]}"
            )
            t1 = time()
            # select data
            logger.info("Selecting data and loading template values")
            selection_data = (
                region_mask_data
                & (dat["Z_not4clus"] > z_range[0])
                & (dat["Z_not4clus"] < z_range[1])
            )
            selected_data = dat[selection_data]

            # don't select randoms further because we're not using the clustering catalogs anyways

            # get data imaging systematics
            data_templates_values = densvar.read_systematic_templates_stacked_alt(
                selected_data["RA"],
                selected_data["DEC"],
                sys_tab,
                fit_maps,
            )

            # add weights
            datacols = list(selected_data.dtype.names)
            logger.info(f"Found columns {cols}")
            logger.info("Using 1/FRACZ_TILELOCID based completeness weights")
            wts = 1 / selected_data["FRACZ_TILELOCID"]
            if "FRAC_TLOBS_TILES" in datacols:
                logger.info("Using FRAC_TLOBS_TILES")
                wts *= 1 / selected_data["FRAC_TLOBS_TILES"]
            else:
                logger.info("no FRAC_TLOBS_TILES")
            if "WEIGHT_ZFAIL" in datacols:
                logger.info("Using redshift failure weights")
                wts *= selected_data["WEIGHT_ZFAIL"]
            else:
                logger.info("no redshift failure weights")

            data_we = jax.numpy.array(wts)
            rand_we = jax.numpy.ones_like(region_randoms, dtype=float)

            logger.info("Starting regression...")
            regressor = LinearRegressor.from_stacked_templates(
                data_weights=data_we,
                random_weights=rand_we,
                template_values_data=data_templates_values,
                template_values_randoms=randoms_templates_values,
                template_names=fit_maps,
                loglevel=loglevel,
            )
            regressor.cut_outliers(tail=tail)
            regressor.prepare(nbins=nbins)
            optimized_parameters = regressor.regress_minuit()

            logger.info("Regression done!")
            logger.info(f"Optimized parameters are {optimized_parameters}")

            output_loc = (
                output_directory
                / f"{tracer_type}_{region}_{z_range[0]:.1f}_{z_range[1]:.1f}_linfitparam_jax.txt"
            )
            logger.info(f"Writing to {output_loc}", logger)
            with open(output_loc, "w") as fo:
                for par_name, par_value in optimized_parameters.items():
                    fo.write(str(par_name) + " " + str(par_value) + "\n")

            if save_summary_plots:
                figname = (
                    output_directory
                    / f"{tracer_type}_{region}_{z_range[0]:.1f}_{z_range[1]:.1f}_linimsysfit_jax.png"
                )
                logger.info(f"Saving figure to {figname}", logger)
                fig, axes = regressor.plot_overdensity(ylim=[0.7, 1.3])
                fig.savefig(figname)

            weights_imlin[selection_data] = regressor.export_weights()
            t2 = time()
            logger.info(
                f"Done with region {region} and redshift bin {z_range[0]} < z < {z_range[1]}, took {t2 - t1} seconds"
            )

    time_end = time()
    logger.info(
        "All linear regressions are done, took %i seconds. Now writing to disk",
        int(time_end - time_start),
    )

    all_data[output_column_name][data_selection] = weights_imlin
    common.write_LSS_scratchcp(dat, data_catalog_path, logger=logger)
