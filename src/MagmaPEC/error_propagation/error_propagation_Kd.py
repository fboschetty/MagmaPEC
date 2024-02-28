import pandas as pd
import numpy as np

from MagmaPandas.configuration import configuration
from MagmaPandas.MagmaFrames import Melt

KD_Blundy_errors = pd.Series({6: 0.019, 9: 0.04, 100: 0.063})

def get_KD_Bludy_error(melt_composition):

    axis = [0,1][isinstance(melt_composition, pd.DataFrame)]
    alkalis = melt_composition[["Na2O", "K2O"]].sum(axis=axis)

    if isinstance(alkalis, (int, float)):

        composition_filter = np.less(alkalis, KD_Blundy_errors.index.values)
        idx = KD_Blundy_errors.index.values[composition_filter][0]
        
        return KD_Blundy_errors[idx]

    errors = pd.Series(index=melt_composition.index)

    for i, a in alkalis.items():
        composition_filter = np.less(a, KD_Blundy_errors.index.values)
        idx = KD_Blundy_errors.index.values[composition_filter][0]
        errors.loc[i] = KD_Blundy_errors[idx]

    return errors


def get_KD_toplis_error(*args, **kwargs):

    return 0.02

def get_KD_error(melt_composition: Melt, **kwargs):

    model = kwargs.get(
        "model", getattr(configuration, "Kd_model"))
    
    error = {"toplis": get_KD_toplis_error, "blundy": get_KD_Bludy_error}[model]
    
    return error(melt_composition)
