"""
Module for PEC correction of olivine-hosted melt inclusions, with errors on models and input parameters propagated in a Monte Carlo simulation. Includes error propagation for:

- melt composition,
- olivine composition,
- initial MI FeO content,
- melt |Fe3Fe2| ratios and
- olivine-melt Fe-Mg partition coefficients.
"""

from MagmaPEC.error_propagation.FeOi_error_propagation import FeOi_prediction
from MagmaPEC.error_propagation.MC_parameters import PEC_MC_parameters
from MagmaPEC.error_propagation.pec_MC_model import PEC_MC
