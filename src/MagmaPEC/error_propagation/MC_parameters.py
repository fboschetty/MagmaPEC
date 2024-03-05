from collections import OrderedDict

import MagmaPandas as mp
import numpy as np
import pandas as pd
from MagmaPandas.Fe_redox import Fe3Fe2_models
from MagmaPandas.Kd.Ol_melt import Kd_models

from .FeOi_error_propagation import FeOi_prediction


class PEC_MC_parameters:

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
