from typing import Tuple, Union

import numpy as np
import pandas as pd
from MagmaPandas.MagmaFrames import Melt, Olivine

from MagmaPEC.PEC_configuration import PEC_configuration
from MagmaPEC.PEC_model.scalar import (
    crystallisation_correction_scalar,
    equilibration_scalar,
)
from MagmaPEC.PEC_model.vector import crystallisation_correction, equilibration
from MagmaPEC.tools import FeO_Target


class PEC:
    # TODO link to configuration documentation.
    """
    Class for post-entrapment crystallisation (PEC) correction of olivine-hosted melt inclusions.

    Model settings are controlled by the global PEC configuration.

    Parameters
    ----------
    inclusions : :py:class:`~magmapandas:MagmaPandas.MagmaFrames.melt.Melt`
        melt inclusion compositions in oxide wt. %
    olivines : :py:class:`~magmapandas:MagmaPandas.MagmaFrames.olivine.Olivine`, :py:class:`Pandas Series <pandas:pandas.Series>`, float
        Olivine compositions in oxide wt. % as Olivine MagmaFrame, or olivine forsterite contents as pandas Series or float.
    P_bar : float, :py:class:`Pandas Series <pandas:pandas.Series>`
        Pressures in bar
    FeO_target : float, :py:class:`Pandas Series <pandas:pandas.Series>`, Callable
        Melt inclusion initial FeO content as a fixed value for all inclusions, inclusion specific values, or a predictive equation based on melt composition. The callable needs to accept a :py:class:`Pandas DataFrame <pandas:pandas.DataFrame>` with melt compositions in oxide wt. % as input and return an array-like object with initial FeO contents per inclusion.
    Fe3Fe2_offset_parameters : float, array-like
        offsets of calculated melt Fe3+/Fe2+ ratios in standard deviations.
    Kd_offset_parameters : float, array-like
        offsets of calculated olivine-melt Fe-Mg Kd's in standard deviations.
    temperature_offset_parameters : float, array-like
        offsets of calculated temperatures in standard deviations.

    Attributes
    ----------
    inclusions : :py:class:`~magmapandas:MagmaPandas.MagmaFrames.melt.Melt`
        Melt inclusion compositions in oxide wt. %. Compositions are updated during the PEC correction procedure
    P_bar : :py:class:`Pandas Series <pandas:pandas.Series>`
        Pressures in bar
    FeO_target : float, :py:class:`Pandas Series <pandas:pandas.Series>`, Callable
        Melt inclusion initial FeO contents.
    olivine_corrected : :py:class:`Pandas DataFrame <pandas:pandas.DataFrame>`
        Dataframe with columns *equilibration_crystallisation*, *PE_crystallisation* and *total_crystallisation*, storing total PEC and crystallisation amounts during the equilibration and crystallisation stages.


    """

    def __init__(
        self,
        inclusions: Melt,
        olivines: Union[Olivine, pd.Series, float],
        P_bar: Union[float, int, pd.Series],
        FeO_target: Union[float, pd.Series, callable],
        Fe3Fe2_offset_parameters: float = 0.0,
        Kd_offset_parameters: float = 0.0,
        temperature_offset_parameters: float = 0.0,
        **kwargs,
    ):

        self.offset_parameters = {
            "Fe3Fe2": Fe3Fe2_offset_parameters,
            "Kd": Kd_offset_parameters,
            "temperature": temperature_offset_parameters,
        }

        self._olivine_corrected = pd.DataFrame(
            0.0,
            columns=[
                "equilibration_crystallisation",
                "PE_crystallisation",
                "total_crystallisation",
            ],
            index=inclusions.index,
        )
        self._FeO_as_function = False

        # Process attributes
        ######################

        # For inclusions
        if not isinstance(inclusions, Melt):
            raise TypeError("Inclusions is not a Melt MagmaFrame")
        else:
            inclusions = inclusions.fillna(0.0)
            self.inclusions = inclusions.normalise()
            self.inclusions_uncorrected = self.inclusions.copy()
        # For olivine
        if hasattr(olivines, "index"):
            try:
                olivines = olivines.loc[inclusions.index]
            except KeyError as err:
                print("Inclusion and olivine indeces don't match")
                raise err

        # For olivines
        if not isinstance(olivines, Olivine):
            try:
                if len(olivines) != self.inclusions.shape[0]:
                    raise ValueError("Number of olivines and inclusions does not match")
            except TypeError:
                pass
            forsterite = pd.Series(
                olivines, index=self.inclusions.index, name="forsterite"
            )
            if (~forsterite.between(0, 1)).any():
                raise ValueError(
                    "olivine host forsterite contents are not all between 0 and 1"
                )
            olivine = Olivine(
                {"MgO": forsterite * 2, "FeO": (1 - forsterite) * 2, "SiO2": 1},
                index=self.inclusions.index,
                units="mol fraction",
                datatype="oxide",
            )
            self._olivine = olivine.normalise()
        else:
            olivines = olivines.fillna(0.0)
            self._olivine = olivines.moles
        self._olivine = self._olivine.reindex(
            columns=self.inclusions.columns, fill_value=0.0
        )

        # For pressure
        try:
            if len(P_bar) != self.inclusions.shape[0]:
                raise ValueError(
                    "Number of pressure inputs and melt inclusions does not match"
                )
        except TypeError:
            pass
        if hasattr(P_bar, "index"):
            try:
                P_bar = P_bar.loc[inclusions.index]
            except KeyError as err:
                print("Inclusion and P_bar indeces don't match")
                raise err
            self.P_bar = P_bar
        else:
            self.P_bar = pd.Series(P_bar, index=self.inclusions.index, name=P_bar)

        self.FeO_target = FeO_Target(
            FeO_target=FeO_target, samples=self.inclusions.index
        )

        self._model_results = pd.DataFrame(
            {
                "isothermal_equilibration": pd.NA,
                "Kd_equilibration": pd.NA,
                "FeO_converge": pd.NA,
            },
            index=self.inclusions.index,
            dtype="boolean",
        )

    def reset(self):
        self.inclusions = self.inclusions_uncorrected.copy()
        self._olivine_corrected.loc[:, :] = 0.0
        self._model_results.loc[:, :] = pd.NA

    @property
    def olivine_corrected(self):
        return self._olivine_corrected.mul(100)

    @olivine_corrected.setter
    def olivine_corrected(self, value):
        print("read only!")

    @property
    def olivine(self):
        return self._olivine.wt_pc

    @olivine.setter
    def olivine(self, value):
        print("Olivine is read only")

    @property
    def equilibrated(self) -> pd.Series:
        """
        Booleans indicating which inclusions are in equilibrium with their host olivine
        based on modelled and observed Fe-Mg exchange Kd's.
        """

        Kd_converge = getattr(PEC_configuration, "Kd_converge") * 1.5
        Kd_equilibrium, Kd_real = self.calculate_Kds()

        return pd.Series(
            np.isclose(Kd_equilibrium, Kd_real, atol=Kd_converge, rtol=0),
            index=Kd_equilibrium.index,
            name="equilibrated",
        )

    @property
    def Fe_loss(self) -> pd.Series:
        """
        Booleans indicating which inclusions have experienced Fe loss
        """
        FeO_converge = getattr(PEC_configuration, "FeO_converge")
        FeO_target = self.FeO_target.target(melt_wtpc=self.inclusions)
        return pd.Series(
            ~np.isclose(
                FeO_target,
                self.inclusions["FeO"],
                atol=FeO_converge,
                rtol=0,
            ),
            index=self.inclusions.index,
            name="Fe_loss",
        )

    def equilibrate_inclusions(self, **kwargs):

        model = equilibration(
            inclusions=self.inclusions,
            olivines=self._olivine,
            P_bar=self.P_bar,
            offset_parameters=self.offset_parameters,
        )

        corrected_melt_compositions, olivine_crystallised, model_results = (
            model.equilibrate(inplace=False, **kwargs)
        )

        self.inclusions = corrected_melt_compositions
        self._olivine_corrected["equilibration_crystallisation"] = (
            olivine_crystallised.values
        )
        self._model_results["isothermal_equilibration"] = model_results.values

    def correct_olivine_crystallisation(self, inplace=False, **kwargs):
        """
        Correct an olivine hosted melt inclusion for post entrapment crystallisation or melting by
        respectively melting or crystallising host olivine.
        Expects the melt inclusion is completely equilibrated with the host crystal.
        The models exits when the user input original melt inclusion FeO content is reached.
        Loosely based on the postentrapment reequilibration procedure in Petrolog:

        L. V. Danyushesky and P. Plechov (2011)
        Petrolog3: Integrated software for modeling crystallization processes
        Geochemistry, Geophysics, Geosystems, vol 12
        """

        if not self._model_results["isothermal_equilibration"].any():
            raise RuntimeError("None of the inclusions are equilibrated")

        # only select samples that reached isothermal equilibrium.
        select_samples = self._model_results["isothermal_equilibration"]

        model = crystallisation_correction(
            inclusions=self.inclusions.loc[select_samples],
            olivines=self._olivine.loc[select_samples],
            P_bar=self.P_bar.loc[select_samples],
            FeO_target=self.FeO_target,
            offset_parameters=self.offset_parameters,
        )

        corrected_melt_compositions, olivine_crystallised, model_results = (
            model.correct(
                equilibration_crystallisation=self._olivine_corrected.loc[
                    select_samples, "equilibration_crystallisation"
                ],
                inplace=False,
            )
        )

        samples = corrected_melt_compositions.index
        self.inclusions.loc[samples] = corrected_melt_compositions.values
        self._olivine_corrected.loc[samples, "PE_crystallisation"] = (
            olivine_crystallised.values
        )
        self._model_results.loc[samples, ["Kd_equilibration", "FeO_converge"]] = (
            model_results.values
        )

    def correct(self, **kwargs) -> Tuple[Melt, pd.DataFrame, pd.DataFrame]:
        """
        Correct inclusions for PEC
        """

        self.equilibrate_inclusions(**kwargs)
        self.correct_olivine_crystallisation(**kwargs)

        self._olivine_corrected["total_crystallisation"] = self._olivine_corrected[
            ["equilibration_crystallisation", "PE_crystallisation"]
        ].sum(axis=1)

        return (
            self.inclusions.copy(),
            self._olivine_corrected.mul(100),
            self._model_results.copy(),
        )

    def correct_inclusion(self, index, plot=True, **kwargs) -> pd.DataFrame:
        """Correct a single inclusion for PEC"""

        if type(index) == int:
            index = self.inclusions_uncorrected.index[index]

        inclusion = self.inclusions_uncorrected.loc[index].copy().squeeze()
        olivine = self._olivine.loc[index].copy().squeeze()
        FeO_target = self.FeO_target.target(melt_wtpc=inclusion)
        P_bar = self.P_bar.loc[index].squeeze()

        if self.FeO_target._as_function:
            FeO_target = self.FeO_target.function

        equilibrated, olivine_equilibrated, *_ = equilibration_scalar(
            inclusion, olivine, P_bar, **kwargs
        )
        corrected, olivine_corrected, *_ = crystallisation_correction_scalar(
            equilibrated.iloc[-1].copy(),
            olivine,
            FeO_target,
            P_bar,
            eq_crystallised=olivine_equilibrated[-1],
            **kwargs,
        )
        total_corrected = olivine_corrected[-1]

        equilibrated["correction"] = "equilibration"
        corrected["correction"] = "correction"

        total_inclusion = pd.concat([equilibrated, corrected.iloc[1:]], axis=0)

        if plot:
            import matplotlib.lines as l
            import matplotlib.pyplot as plt
            from labellines import labelLine

            set_markers = kwargs.get("markers", True)

            import geoplot as gp

            fontsize = 14

            fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=False)
            plt.grid(True, color="whitesmoke")

            colors = gp.colors.flatDesign.by_key()["color"]

            linewidth = 5
            markersize = 90

            FeO_color = tuple(np.repeat(0.25, 3))

            plt.plot(
                equilibrated["MgO"],
                equilibrated["FeO"],
                ["-", ".-"][set_markers],
                color=colors[1],
                # label="equilibration",
                linewidth=linewidth,
                mec="k",
                markersize=10,
                # alpha=0.7,
            )
            plt.plot(
                corrected["MgO"],
                corrected["FeO"],
                ["-", ".-"][set_markers],
                color=colors[2],
                linewidth=linewidth,
                mec="k",
                markersize=10,
                # alpha=0.7,
            )
            ax.scatter(
                equilibrated.loc[equilibrated.index[0], "MgO"],
                equilibrated.loc[equilibrated.index[0], "FeO"],
                marker="^",
                color=colors[1],
                edgecolors="k",
                s=markersize,
                zorder=10,
                label="Glass",
            )
            ax.scatter(
                equilibrated.loc[equilibrated.index[-1], "MgO"],
                equilibrated.loc[equilibrated.index[-1], "FeO"],
                marker="o",
                edgecolors="k",
                color=colors[3],
                s=markersize,
                zorder=10,
                label="Equilibrated",
            )
            ax.scatter(
                corrected.loc[corrected.index[-1], "MgO"],
                corrected.loc[corrected.index[-1], "FeO"],
                marker="s",
                color=colors[2],
                edgecolors="k",
                s=markersize,
                zorder=10,
                label="Corrected",
            )

            middle = sum(ax.get_xlim()) / 2

            if self._FeO_as_function:
                FeO_inital = self.FeO_target.target(melt_wtpc=corrected)
                ax.plot(
                    corrected["MgO"],
                    FeO_inital,
                    "-",
                    color=FeO_color,
                )
                FeO_target = sum((min(FeO_inital), max(FeO_inital))) / 2
            else:
                ax.axhline(FeO_target, linestyle="-", color=FeO_color, linewidth=1.5)

            FeO_line = ax.get_lines()[-1]
            try:
                labelLine(
                    FeO_line,
                    x=middle,
                    label="initial FeO",
                    size=fontsize * 0.8,
                    color=FeO_color,
                )
            except ValueError:
                pass

            ax.set_ylim(ax.get_ylim()[0], max((FeO_target * 1.03, ax.get_ylim()[1])))

            ax.set_xlabel("MgO (wt. %)")
            ax.set_ylabel("FeO$^T$\n(wt. %)", rotation=0, labelpad=30)

            handles, labels = ax.get_legend_handles_labels()

            handles = handles + [l.Line2D([0], [0], linewidth=0)]

            labels = labels + [f"{total_corrected:.2f} mol. %\nPEC correction"]

            ax.legend(
                handles,
                labels,
                title=index,
                prop={"family": "monospace", "size": fontsize / 1.5},
                fancybox=False,
                facecolor="white",
            )

        return total_inclusion
