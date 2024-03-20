from collections import OrderedDict
from typing import Dict

import MagmaPandas as mp
import numpy as np
import pandas as pd
from MagmaPandas.Fe_redox import Fe3Fe2_models
from MagmaPandas.Kd.Ol_melt import Kd_models

from MagmaPEC.error_propagation.FeOi_error_propagation import FeOi_prediction


class PEC_MC_parameters:
    """
    Class for fetching parameters for PEC Monte Carlo simulations.
    Includes error propagation for:

    - melt composition
    - olivine composition
    - initial MI FeO content
    - melt |Fe3Fe2| ratios
    - olivine-melt Fe-Mg partition coefficients.

    Parameters
    ----------
    melt_errors : pandas Series, None
        one standard deviation errors on melt compositions in oxide wt. %. Melt compositions are assumed to be error-free when set to None. Default value: None
    olivine_errors : pandas Series, None
        one standard deviation errors on olivine compositions in oxide wt. %. Melt compositions are assumed to be error-free when set to None. Default value: None
    FeOi_errors : float, pandas Series, :py:class:`~MagmaPEC.error_propagation.FeOi_prediction`
        errors on melt initial FeO content. float or Series for errors on FeO, FeOi_prediction for errors on coefficients of linear regressions of FeO against melt major element compositions. Default value: 0.0
    Fe3Fe2 : bool
        propagate melt |Fe3Fe2| errors. Default value: False
    Kd : bool
        propagate olivine-melt Fe-Mg partition coefficient errors. Default value: False

    Attributes
    ----------
    parameters : Dict
    """

    parameters = OrderedDict()

    def __init__(
        self,
        melt_errors: None | pd.Series = None,
        olivine_errors: None | pd.Series = None,
        FeOi_errors: float | FeOi_prediction = 0.0,
        Fe3Fe2: bool = False,
        Kd: bool = False,
    ):
        self.melt_errors = melt_errors
        self.olivine_errors = olivine_errors
        self.FeOi_errors = FeOi_errors
        self.Fe3Fe2 = Fe3Fe2
        self.Kd = Kd

    def get_parameters(self, n: int):
        """
        Randomly sample parameter errors. Results are stored in ``parameters``.

        melt, olivine and FeOi errors are calculated based on user input values. |Fe3Fe2| and Kd errors are in standard deviations and calculated based on model calibration errors.

        Parameters
        ----------
        n : int
            amount of random samples.
        """

        Fe3Fe2_model = Fe3Fe2_models[mp.configuration.Fe3Fe2_model]
        Kd_model = Kd_models[mp.configuration.Kd_model]

        if self.melt_errors is None:
            self.parameters["melt"] = np.repeat(0.0, n)
        else:
            self.parameters["melt"] = pd.DataFrame(
                np.random.normal(
                    loc=0, scale=self.melt_errors, size=(n, len(self.melt_errors))
                ),
                columns=self.melt_errors.index,
            )

        if self.olivine_errors is None:
            self.parameters["olivine"] = np.repeat(0.0, n)
        else:
            self.parameters["olivine"] = pd.DataFrame(
                np.random.normal(
                    loc=0, scale=self.olivine_errors, size=(n, len(self.olivine_errors))
                ),
                columns=self.olivine_errors.index,
            )

        if isinstance(self.FeOi_errors, (float, int)):
            self.parameters["FeOi"] = np.random.normal(
                loc=0, scale=self.FeOi_errors, size=n
            )
        elif isinstance(self.FeOi_errors, pd.Series):
            self.parameters["FeOi"] = pd.DataFrame(
                np.random.normal(
                    loc=0, scale=self.FeOi_errors, size=(n, len(self.FeOi_errors))
                ),
                columns=self.FeOi_errors.index,
            )
        elif isinstance(self.FeOi_errors, FeOi_prediction):
            self.parameters["FeOi"] = self.FeOi_errors.random_sample_coefficients(n=n)

        if self.Fe3Fe2:
            self.parameters["Fe3Fe2"] = Fe3Fe2_model.get_offset_parameters(n=n)
        else:
            self.parameters["Fe3Fe2"] = np.repeat(0.0, n)

        if self.Kd:
            self.parameters["Kd"] = Kd_model.get_offset_parameters(n=n)
        else:
            self.parameters["Kd"] = np.repeat(0.0, n)

    def _get_iterators(self):
        return [
            i.iterrows() if isinstance(i, pd.DataFrame) else i
            for i in self.parameters.values()
        ]
