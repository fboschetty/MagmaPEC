import statsmodels.api as sm
import pandas as pd
import numpy as np

def get_linear_regression_coeffictients(x, y):

    x_train = sm.add_constant(x)
    reg_ols = sm.OLS(y, x_train).fit()

    intersection, slope = reg_ols.params
    intersection_err, slope_err = reg_ols.bse

    return (slope, intersection), (slope_err, intersection_err)

def linear_regression_coefficients_MC(x, y, n=50):

    (slope, _) , (slope_err, _) = get_linear_regression_coeffictients(x, y)

    MC_slopes = np.random.normal(loc=slope, scale=slope_err, size=n)

    coefficients = pd.DataFrame({"slope": MC_slopes}, index=range(n), dtype=np.float16)
    coefficients["intercept"] = (y.mean() - MC_slopes * x.mean()).astype(np.float16)

    return coefficients.squeeze()
