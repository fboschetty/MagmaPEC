from MagmaPandas.configuration import configuration
from MagmaPandas.Fe_redox import Fe3Fe2_models
from MagmaPandas.fO2 import calculate_fO2
from MagmaPandas.Kd.Ol_melt.models import Kd_models


def _root_temperature(
    olivine_amount, melt_x_moles, olivine_x_moles, T_K, P_bar, temperature_offset=0.0
):

    melt_x_new = melt_x_moles + olivine_x_moles.mul(olivine_amount)
    melt_x_new = melt_x_new.normalise()
    temperature_new = melt_x_new.temperature(P_bar=P_bar, offset=temperature_offset)

    return T_K - temperature_new


def _root_Kd(exchange_amount, melt_x_moles, exchange_vector, forsterite, P_bar, kwargs):

    melt_x_new = melt_x_moles + exchange_vector.mul(exchange_amount)
    melt_x_new = melt_x_new.normalise()
    Kd_equilibrium, Kd_real = calculate_Kds(melt_x_new, P_bar, forsterite, **kwargs)

    return Kd_equilibrium - Kd_real


def calculate_Kds(
    melt_x_moles,
    P_bar,
    forsterite,
    offset_parameters=0.0,
    temperature_offset=0.0,
    **kwargs,
):

    Fe3Fe2_model = Fe3Fe2_models[configuration.Fe3Fe2_model]
    Kd_model = Kd_models[configuration.Kd_model]
    dfO2 = kwargs.get("dfO2", configuration.dfO2)

    T_K = melt_x_moles.temperature(P_bar, offset=temperature_offset)
    fO2 = calculate_fO2(T_K=T_K, P_bar=P_bar, dfO2=dfO2)
    Fe3Fe2 = Fe3Fe2_model.calculate_Fe3Fe2(
        melt_mol_fractions=melt_x_moles, P_bar=P_bar, T_K=T_K, fO2=fO2
    )

    Kd_observed = calculate_observed_Kd(melt_x_moles, Fe3Fe2, forsterite)

    Kd_eq = Kd_model.calculate_Kd(
        melt_mol_fractions=melt_x_moles,
        forsterite_initial=forsterite,
        T_K=T_K,
        Fe3Fe2=Fe3Fe2,
        P_bar=P_bar,
    )
    Kd_eq = Kd_eq + offset_parameters
    Kd_eq[Kd_eq <= 0.0] = 1e-6

    return Kd_eq, Kd_observed


def calculate_observed_Kd(melt_mol_fractions, Fe3Fe2, forsterite):

    melt_mol_fractions = melt_mol_fractions.normalise()
    Fe2_FeTotal = 1 / (1 + Fe3Fe2)
    melt_MgFe = melt_mol_fractions["MgO"] / (melt_mol_fractions["FeO"] * Fe2_FeTotal)
    olivine_MgFe = forsterite / (1 - forsterite)

    return melt_MgFe / olivine_MgFe
