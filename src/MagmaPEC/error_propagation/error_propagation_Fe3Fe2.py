from MagmaPandas.configuration import configuration
from MagmaPandas.MagmaFrames import Melt


def Fe3Fe2_Borisov_error(melt_composition: Melt, T_K, P_bar):
    """
    propagate log(Fe3/Fe2) error to Fe3/Fe2
    """

    Fe3Fe2 = melt_composition.Fe3Fe2_QFM(T_K=T_K, P_bar=P_bar, Fe_model="borisov")
    
    return 0.079 * Fe3Fe2

KressCarmichael_FeO_error = 0.21
KressCarmichael_Fe2O3_error = 0.42

def Fe3Fe2_CressCarmichael_error(melt_composition: Melt, T_K, P_bar):
    """
    Propagate Fe2O3 and FeO errors to Fe3/Fe2
    """
    
    Fe3Fe2 = melt_composition.Fe3Fe2_QFM(T_K=T_K, P_bar=P_bar, Fe_model="kressCarmichael")
    melt = melt_composition.FeO_Fe2O3_calc(Fe3Fe2, total_Fe="FeO", inplace=False)

    Fe2O3_contribution = KressCarmichael_Fe2O3_error / 2 / melt["Fe2O3"]
    FeO_contribution = KressCarmichael_FeO_error / melt["FeO"]

    return (Fe2O3_contribution + FeO_contribution) * Fe3Fe2


def get_Fe3Fe2_error(melt_composition, T_K, P_bar, **kwargs):

    model = kwargs.get(
        "model", getattr(configuration, "Fe3Fe2_model"))

    func = {"borisov": Fe3Fe2_Borisov_error, "kressCarmichael": Fe3Fe2_CressCarmichael_error}[model]

    return func(melt_composition, T_K=T_K, P_bar=P_bar)
