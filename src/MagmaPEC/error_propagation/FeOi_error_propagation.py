import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_validate
from statsmodels.tools.eval_measures import rmse

# def get_linear_regression_coeffictients(x, y):

#     x_train = sm.add_constant(x)
#     reg_ols = sm.OLS(y, x_train).fit()

#     intersection, slope = reg_ols.params
#     intersection_err, slope_err = reg_ols.bse

#     return (slope, intersection), (slope_err, intersection_err)


# def linear_regression_coefficients_MC(x, y, n=1):

#     (slope, _), (slope_err, _) = get_linear_regression_coeffictients(x, y)

#     MC_slopes = np.random.normal(loc=slope, scale=slope_err, size=n)

#     coefficients = pd.DataFrame({"slope": MC_slopes}, index=range(n), dtype=np.float16)
#     coefficients["intercept"] = (y.mean() - MC_slopes * x.mean()).astype(np.float16)

#     return coefficients.squeeze()


class FeOi_prediction:

    def __init__(self, x, y):

        self.x = x
        self.y = y

        self.predictors = x.columns.values

    def get_OLS_coefficients(self) -> None:

        reg_ols = self._OLS_fit()

        self.slopes = reg_ols.params
        self.slopes_error = reg_ols.bse

        self.intercept = self.slopes.pop("const")
        self.intercept_error = self.slopes_error.pop("const")

    def random_sample_coefficients(self, n) -> None:

        if not hasattr(self, "errors"):
            self.get_OLS_coefficients()

        coeff_MC = pd.DataFrame(
            np.random.normal(
                loc=self.slopes, scale=self.slopes_error, size=(n, len(self.slopes))
            ),
            columns=self.slopes.index,
            dtype=np.float16,
        )

        coeff_MC["intercept"] = (
            self.y.mean() - coeff_MC.mul(self.x.mean(), axis=1).sum(axis=1)
        ).astype(np.float16)
        # coeff_MC = coeff_MC.squeeze()

        return coeff_MC

    def _OLS_fit(self, x=None, y=None) -> sm.regression.linear_model.RegressionResults:

        if x is None:
            x = self.x[self.predictors]
        if y is None:
            y = self.y

        x_train = sm.add_constant(x)

        return sm.OLS(y, x_train).fit()

    def _f_test(
        self,
        data: pd.DataFrame,
        regression1: sm.regression.linear_model.RegressionResults,
        regression2: sm.regression.linear_model.RegressionResults,
    ) -> float:
        """
        F test for models comparison
        """
        return (
            abs((regression1.resid**2).sum() - (regression2.resid**2).sum())
            / (len(regression2.params) - len(regression1.params))
            / ((regression2.resid**2).sum() / (len(data) - len(regression2.params)))
        )

    def _f_to_p(
        self,
        data: pd.DataFrame,
        regression1: sm.regression.linear_model.RegressionResults,
        regression2: sm.regression.linear_model.RegressionResults,
    ) -> float:
        """
        convert F-value to associated P-value
        """
        f = self._f_test(data, regression1, regression2)
        return stats.f.cdf(
            f,
            abs(len(regression1.params) - len(regression2.params)),
            len(data) - len(regression2.params),
        )

    def _compare_models(self, x, y):
        """
        Regress y on x-1 and compare with regressions of y on x
        """
        # Regress y on x
        upper_reg = self._OLS_fit(x, y)

        results = pd.DataFrame(
            columns=["f_ratio", "p_value", "r_ratio"],
            index=np.arange(0, x.shape[1], 1),
            dtype=float,
        )
        parameters = {i: None for i in results.index}
        models = {i: None for i in results.index}

        # Regress y on x-1 for all permutations
        for col_drop in results.index:
            x_reduced = x.drop(x.columns[col_drop], axis=1)
            models[col_drop] = self._OLS_fit(x_reduced, y)
            # Compare with y regressed on x
            results.loc[col_drop, "f_ratio"] = self._f_test(
                x, models[col_drop], upper_reg
            )
            results.loc[col_drop, "p_value"] = self._f_to_p(
                x, models[col_drop], upper_reg
            )
            results.loc[col_drop, "r2"] = models[col_drop].rsquared

            parameters[col_drop] = pd.concat(
                [models[col_drop].params, models[col_drop].pvalues], axis=1
            )
            parameters[col_drop].columns = ["coefficients", "p_value"]

        return results, parameters, models

    def _get_model(self, x, y):
        """
        Compare regression results and select the statistically best model
        """

        # Get models
        results, parameters, models = self._compare_models(x, y)

        # Select the model with the lowest p vaue
        max_index = results["p_value"].idxmin()
        x_data = x.drop(x.columns[max_index], axis=1)
        return (
            x_data,
            parameters[max_index],
            models[max_index],
        )

    def calculate_model_fits(self, exclude=None, crossvalidation_split=0.15):

        x = self.x.copy()
        if exclude is not None:
            x = x.drop(columns=exclude)

        y = self.y.copy()

        parameters_total = pd.DataFrame(
            dtype=float, columns=["const"] + list(x.columns)
        )
        self.model_fits = pd.DataFrame(
            dtype=float, columns=["calibration", "validation", "delta", "r2"]
        )

        for _ in range(x.shape[1] - 1):  # x.columns[1:]
            x_data, parameters, model = self._get_model(x, y)

            # Copy coefficients
            n_parameters = len(x_data.columns)
            parameters_total.loc[n_parameters, parameters.index] = parameters[
                "coefficients"
            ]

            # Calculate errors
            self.model_fits.loc[n_parameters, "calibration"] = rmse(
                y, model.predict(sm.add_constant(x_data))
            )
            cross_validation = cross_validate(
                statsmodel_wrapper(sm.OLS),  # LinearRegression(),
                x_data,
                y,
                cv=int(1 / crossvalidation_split),
                scoring=("neg_mean_squared_error"),
            )
            self.model_fits.loc[n_parameters, "validation"] = np.sqrt(
                abs(cross_validation["test_score"])
            ).mean()

            self.model_fits.loc[n_parameters, "r2"] = model.rsquared
            x = x_data.copy()

        self.model_fits["delta"] = abs(
            self.model_fits["validation"] - self.model_fits["calibration"]
        )
        self.coeff_total = parameters_total.rename(columns={"const": "intercept"})

        print(pd.concat([self.coeff_total, self.model_fits], axis=1))

    def select_predictors(self, idx: int):

        if not hasattr(self, "model_fits"):
            raise AttributeError(
                "model_fits attribute missing, run calculate_model_fits first!"
            )

        self.predictors = (
            self.coeff_total.loc[idx].drop("intercept").dropna().index.values
        )
        self.get_OLS_coefficients()

    @staticmethod
    def _FeO_initial_func(composition, coefficients):
        if isinstance(composition, pd.DataFrame):
            oxides = composition.columns.intersection(coefficients.index)
            return coefficients["intercept"] + composition[oxides].mul(
                coefficients.loc[oxides], axis=1
            ).sum(axis=1).astype(np.float32)
        elif isinstance(composition, pd.Series):
            oxides = composition.index.intersection(coefficients.index)
            return (
                coefficients["intercept"]
                + composition[oxides].mul(coefficients.loc[oxides]).sum()
            ).astype(np.float32)


# A wrapper for statsmodel, for use within sklearn
# Not really needed since LinearRegression produces the same results as sm.OLS
class statsmodel_wrapper(BaseEstimator, RegressorMixin):
    """A universal sklearn-style wrapper for statsmodels regressors"""

    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)
