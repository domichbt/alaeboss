import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import numpy.testing as npt

from .linear_regression import LinearRegressor

jax.config.update("jax_enable_x64", True)


class TestLinearRegressor:
    key1 = jrd.key(1234)
    key2 = jrd.key(5678)
    datalen = int(1e6)
    randomlen = int(1e8)
    nbins = 10

    def test_whole_regression(self):
        expected_res = {
            "constant": -0.9952391264847271,
            "syst1": 0.8370850695798598,
            "syst2": 1.677471769169541,
        }

        # The sky is a beautiful 1-D array with two gaussian hotpoints of systematics
        # The data follows this distribution closely, and we write down the associated values of the systematics at this point
        sky = jnp.linspace(0, 1, 786432)
        sysmap1 = 1 * jnp.exp(-jnp.square((sky - 0.25) / 0.1) / 2)
        sysmap2 = 2 * jnp.exp(-jnp.square((sky - 0.75) / 0.2) / 2)

        data_weights_test = jnp.ones(shape=(self.datalen), dtype=float)
        random_weights_test = jnp.ones(shape=(self.randomlen), dtype=float)

        data_loc = jrd.choice(
            key=self.key1,
            a=jnp.arange(len(sky)),
            p=sysmap1 + sysmap2,
            replace=True,
            shape=(self.datalen,),
        )
        random_loc = jrd.choice(
            key=self.key2,
            a=jnp.arange(len(sky)),
            p=None,
            replace=True,
            shape=(self.randomlen,),
        )

        templates_test = {
            "syst1": (sysmap1[data_loc], sysmap1[random_loc]),
            "syst2": (sysmap2[data_loc], sysmap2[random_loc]),
        }

        r = LinearRegressor(
            data_weights=data_weights_test,
            random_weights=random_weights_test,
            templates=templates_test,
            loglevel="INFO",
        )
        r.cut_outliers(tail=0.5)
        r.prepare(nbins=self.nbins)
        res = r.regress_minuit()
        diff_dict = {name: (res[name] - k2) / k2 for name, k2 in expected_res.items()}

        res_tab = np.array(list(res.values()))
        expected_res_tab = np.array(list(expected_res.values()))

        npt.assert_allclose(
            res_tab,
            expected_res_tab,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Test failed with relative differences for each coefficient: {diff_dict}",
        )
