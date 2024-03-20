from functools import partial

import pandas as pd
from IPython.display import clear_output
from MagmaPandas.MagmaFrames import Melt, Olivine

from ..PEC_model import *
from .FeOi_error_propagation import FeOi_prediction
from .MC_parameters import *


class PEC_MC:

    def __init__(
        self,
        inclusions: Melt,
        olivines: Olivine,
        P_bar: float | pd.Series,
        FeO_target: float | FeOi_prediction,
        MC_parameters: PEC_MC_parameters,
    ):

        self.inclusions = inclusions
        self.olivines = olivines
        self.P_bar = P_bar
        self.FeO_target = FeO_target

        self.parameters = MC_parameters

    def run(self, n: int):

        self.pec_MC = pd.DataFrame(columns=self.inclusions.index, index=range(n))
        self.compositions_MC = {
            name: pd.DataFrame(columns=self.inclusions.columns, index=range(n))
            for name in self.inclusions.index
        }

        self.parameters.get_parameters(n=n)

        for i, (*params, Fe3Fe2_err, Kd_err) in enumerate(
            zip(*self.parameters._get_iterators())
        ):
            clear_output()
            print(f"Monte Carlo loop\n{i+1:03}/{n:03}")

            melt_MC, olivine_MC, FeOi = self._process_MC_params(*params)

            pec_model = PEC_olivine(
                inclusions=melt_MC,
                olivines=olivine_MC,
                P_bar=self.P_bar,
                FeO_target=FeOi,
                Fe3Fe2_offset_parameters=Fe3Fe2_err,
                Kd_offset_parameters=Kd_err,
            )

            melts_corr, pec, T_K = pec_model.correct()
            for name, row in melts_corr.iterrows():
                self.compositions_MC[name].loc[i] = row
                self.pec_MC.loc[i, name] = pec.loc[name, "total_crystallisation"]

        self._calculate_errors()

    def _process_MC_params(self, melt_err, olivine_err, FeOi_err):

        melt_err = melt_err[1] if isinstance(melt_err, tuple) else melt_err
        melt_MC = self.inclusions.add(melt_err, axis=1)
        olivine_err = olivine_err[1] if isinstance(olivine_err, tuple) else olivine_err
        olivine_MC = self.olivines.add(olivine_err, axis=1)

        if isinstance(FeOi_err, (float, int)):
            # for a fixed FeO error
            FeOi = self.FeO_target + FeOi_err
        elif isinstance(FeOi_err, tuple):
            # for errors on linear regression coefficients
            if "intercept" in FeOi_err[1].index:
                FeOi = partial(
                    self.FeO_target._FeO_initial_func, coefficients=FeOi_err[1]
                )
            elif FeOi_err[1].index.equals(self.inclusions.index):
                # for FeO errors per inclusion
                FeOi = self.FeO_target + FeOi_err[1]

        return melt_MC, olivine_MC, FeOi

    def _calculate_errors(self):

        self.pec = pd.concat([self.pec_MC.mean(), self.pec_MC.std()], axis=1)
        self.pec.columns = ("pec", "stddev")

        self.inclusions_corr = pd.DataFrame(
            index=self.inclusions.index, columns=self.inclusions.columns
        )
        colnames = [f"{n}_stddev" for n in self.inclusions.columns]
        self.inclusions_stddev = pd.DataFrame(
            index=self.inclusions.index, columns=colnames
        )

        for inclusion, df in self.compositions_MC.items():
            self.inclusions_corr.loc[inclusion] = df.mean().values
            self.inclusions_stddev.loc[inclusion] = df.std().values

        try:
            self.inclusions_stddev.drop(columns=["total_stddev"], inplace=True)
        except KeyError:
            pass
