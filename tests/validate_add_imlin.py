"""
Script to validate `scripts/add_imlin.py` against the LSS `add_imlin_clus.py`.

This is a python script that calls onto the other two scripts with the same CLI options and compares the final catalog.
It does NOT test any of the options that would overwrite existing catalog columns ("WEIGHT_SYS", "WEIGHT"...) or change the randoms.
"""

import logging
import shlex
import subprocess
from pathlib import Path
from time import time

import numpy as np
from astropy.table import Table

logger = logging.getLogger("Validation")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s [%(name)s] %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
file_handler = logging.FileHandler(
    filename="tests/output/validate_add_imlin.log", mode="w"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

"""
usage: add_imlin.py [-h] [--type TYPE] [--basedir BASEDIR] [--version VERSION]
                    [--survey SURVEY] [--verspec VERSPEC]
                    [--usemaps [USEMAPS ...]] [--exclude_debv]
                    [--relax_zbounds] [--imsys_finezbin] [--imsys_1zbin]
                    [--imsys_clus] [--imsys_clus_ran] [--replace_syscol]
                    [--add_syscol2blind] [--Y1_mode] [--imsys_zbin IMSYS_ZBIN]
                    [--par PAR] [--extra_clus_dir EXTRA_CLUS_DIR]
                    [--minr MINR] [--maxr MAXR] [--syscol SYSCOL]
                    [--nran4imsys NRAN4IMSYS]
"""

LSS_CAT_SCRIPT = "python /global/u1/d/dchebat/gitclones_r/LSS/scripts/add_imlin_clus.py"
JAX_CAT_SCRIPT = "python /global/homes/d/dchebat/imsys/alaeboss/scripts/add_imlin.py"

# cli_args_dict = {
#     "type": None,
#     "basedir": None,
#     "version": None,
#     "survey": None,
#     "verspec": None,
#     "usemaps": None,
#     "relax_zbounds": None,
#     "imsys_finezbin": None,
#     "imsys_1zbin": None,
#     "imsys_clus": None,
#     "imsys_clus_ran": None,
#     "replace_syscol": None,
#     "add_syscol2blind": None,
#     "Y1_mode": None,
#     "imsys_zbin": None,
#     "par": None,
#     "extra_clus_dir": None,
#     "minr": None,
#     "maxr": None,
#     "syscol": None,
#     "nran4imsys": None,
# }

# fmt: off
cli_args_fixed = {
    "--version": "v1.5",
    "--survey": "Y1",
    "--verspec": "iron",
    "--usemaps": False,  # let it be determined automatically
    "--imsys_clus_ran": False,
    "--replace_syscol": False,
    "--add_syscol2blind": False,
    "--par": False,
    "--extra_clus_dir": "dummy",  # otherwise fNL automatically
    "--minr": False,
    "--maxr": False,
    "--syscol": "WEIGHT_COMPARE", 
    "--nran4imsys": "1",  # for quick tests
    "--relax_zbounds": False,  # does nothing anyways
    "--imsys_clus": True,  # Has to happen!
}
# fmt: off

basedir_LSS = "/pscratch/sd/d/dchebat/imsys_fastload_LSS"
basedir_JAX = "/pscratch/sd/d/dchebat/imsys_fastload_JAX"

cli_args_list_fixed = []
for k, v in cli_args_fixed.items():
    if isinstance(v, bool):
        if v:
            cli_args_list_fixed.append(k)
    else:
        cli_args_list_fixed.append(k)
        cli_args_list_fixed.append(v)

# have to pick one of three between imsys_zbin, imsys_finezbin and imsys_1zbin
redshift_binnings = ["--imsys_zbin y", "--imsys_finezbin", "--imsys_1zbin"]
tracer_types = ["LRG", "QSO"]


def run_comparison(fixed_args, redshift_binning, tracer_type, Y1_mode, log_directory):
    log_directory = Path(log_directory)
    file_basis_name = f"{redshift_binning[2:].replace(' ', '=')}-{tracer_type}-Y1={Y1_mode}"

    cli_args = fixed_args + [redshift_binning, "--type", tracer_type]

    if Y1_mode:
        cli_args.append("--Y1_mode")

    cli_args_str = " ".join(cli_args)

    LSS_cmd = shlex.split(
        " ".join([LSS_CAT_SCRIPT, f"--basedir {basedir_LSS}", cli_args_str])
    )
    JAX_cmd = shlex.split(
        " ".join([JAX_CAT_SCRIPT, f"--basedir {basedir_JAX}", cli_args_str])
    )

    logger.info("Starting LSS run...")
    time_start_LSS = time()
    LSS_run = subprocess.run(
        args=LSS_cmd,
        capture_output=True,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
    )
    time_end_LSS = time()
    logger.info("LSS run is done ; took %i seconds", int(time_end_LSS - time_start_LSS))
    logger.info("Arguments were %s", LSS_run.args)
    with open(log_directory/f"LSS-{file_basis_name}.out", 'wb') as file:
        file.write(LSS_run.stdout)
    with open(log_directory/f"LSS-{file_basis_name}.err", 'wb') as file:
        file.write(LSS_run.stderr)

    logger.info("Starting JAX run...")
    time_start_JAX = time()
    JAX_run = subprocess.run(args=JAX_cmd, capture_output=True)
    time_end_JAX = time()
    logger.info("JAX run is done ; took %i seconds", int(time_end_JAX - time_start_JAX))

    with open(log_directory/f"JAX-{file_basis_name}.out", 'wb') as file:
        file.write(JAX_run.stdout)
    with open(log_directory/f"JAX-{file_basis_name}.err", 'wb') as file:
        file.write(JAX_run.stderr)

    logger.info("Wrote STDOUT and STDERR to %s", log_directory)

    logger.info("Can now compare weight columns")
    
    syscol = "WEIGHT_COMPARE"
    try:
        JAX_cat_SGC = Path("/pscratch/sd/d/dchebat/imsys_fastload_JAX/Y1/LSS/iron/LSScats/v1.5/dummy")/f"{tracer_type}_SGC_clustering.dat.fits"
        JAX_cat_NGC = Path("/pscratch/sd/d/dchebat/imsys_fastload_JAX/Y1/LSS/iron/LSScats/v1.5/dummy")/f"{tracer_type}_NGC_clustering.dat.fits"
        SGC_table = Table.read(JAX_cat_SGC)
        NGC_table = Table.read(JAX_cat_NGC)
        JAX_weights = np.concatenate([SGC_table[syscol], NGC_table[syscol]])
        np.save(log_directory/f"JAX-{file_basis_name}", JAX_weights)
        SGC_table.remove_column(syscol)
        NGC_table.remove_column(syscol)
        SGC_table.write(JAX_cat_SGC, overwrite=True)
        NGC_table.write(JAX_cat_NGC, overwrite=True)

        LSS_cat_SGC = Path("/pscratch/sd/d/dchebat/imsys_fastload_LSS/Y1/LSS/iron/LSScats/v1.5/dummy")/f"{tracer_type}_SGC_clustering.dat.fits"
        LSS_cat_NGC = Path("/pscratch/sd/d/dchebat/imsys_fastload_LSS/Y1/LSS/iron/LSScats/v1.5/dummy")/f"{tracer_type}_NGC_clustering.dat.fits"
        SGC_table = Table.read(LSS_cat_SGC)
        NGC_table = Table.read(LSS_cat_NGC)
        LSS_weights = np.concatenate([SGC_table[syscol], NGC_table[syscol]])
        np.save(log_directory/f"LSS-{file_basis_name}", LSS_weights)
        SGC_table.remove_column(syscol)
        NGC_table.remove_column(syscol)
        SGC_table.write(LSS_cat_SGC, overwrite=True)
        NGC_table.write(LSS_cat_NGC, overwrite=True)
        logger.info("Weight columns saved for later perusal. STD of difference is %f", (JAX_weights - LSS_weights).std())

    except KeyError:
        logger.error("Could not find column named WEIGHT_COMPARE. Maybe the scripts failed to run.")


if __name__ == "__main__":
    for tracer_type in tracer_types:
        for redshift_binning in redshift_binnings:
            for Y1_mode in [False, True]:
                # 24 tests in total
                logger.info("Starting comparison for tracer %s, redshift binning %s, Y1_mode set to %s", tracer_type, redshift_binning, Y1_mode)
                run_comparison(
                    fixed_args=cli_args_list_fixed,
                    redshift_binning=redshift_binning,
                    tracer_type=tracer_type,
                    Y1_mode=Y1_mode,
                    log_directory="/global/homes/d/dchebat/imsys/alaeboss/tests/output"
                )
