from .fis import FIS

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    GridSearchCV = None


def _get_vars(fis):
    """Get an encoded version of the parameters of the fuzzy sets in a FIS"""
    for variable in fis.variables:
        for value_name, value in variable.values.items():
            for par, default in value._get_description().items():
                if par != "type":
                    yield "var_" + variable.name + "_" + value_name + "_" + par, default

    # Same for target
    for variable in [fis.target]:  # For symmetry
        for value_name, value in variable.values.items():
            for par, default in value._get_description().items():
                if par != "type":
                    yield "target_" + variable.name + "_" + value_name + "_" + par, default


def _set_vars(fis, kwargs):
    """Return a modified version of the FIS, setting the changes in the parameters described by kwargs"""
    description = fis._get_description()
    positions = {variable["name"]: i for i, variable in enumerate(description["variables"])}
    for code, parameter_value in kwargs.items():
        if code.startswith("var_"):
            _, variable_name, fuzzy_value, parameter = code.split("_")
            description["variables"][positions[variable_name]]["values"][fuzzy_value][parameter] = parameter_value
        elif code.startswith("target_"):
            _, _, fuzzy_value, parameter = code.split("_")
            description["target"]["values"][fuzzy_value][parameter] = parameter_value
        elif code in ["defuzzification"]:
            description[code] = parameter_value
        else:
            raise ValueError("Parameter not supported: %s" % code)

    return FIS._from_description(description)


class ParametrizedFIS:
    """A parametrizable Fuzzy Inference System with a scikit-learn-like interface"""

    def __init__(self, fis, defuzzification="centroid", **kwargs):
        self.fis = fis

        self.defuzzification = defuzzification

        for parameter, value in _get_vars(fis):
            setattr(self, parameter, kwargs.get(parameter, value))

    def get_params(self, deep=True):
        """Get the parameters in a sklearn-consistent interface"""
        return {"fis": self.fis,
                "defuzzification": self.defuzzification,
                **{parameter: getattr(self, parameter) for parameter, _ in _get_vars(self.fis)}}

    def set_params(self, **parameters):
        """Set the parameters in a sklearn-consistent interface"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X=None, y=None):
        """'Fit' the model (freeze the attributes and compile if available)"""
        self.fis_ = _set_vars(self.fis, {parameter: getattr(self, parameter) for parameter, _ in _get_vars(self.fis)})
        self.fis_.defuzzification = self.defuzzification

        try:
            self.fis_ = self.fis_.compile()
        except Exception:
            pass

        return self

    def predict(self, X):
        """A sklearn-like predict method"""
        return self.fis_.batch_predict(X)


class TrivialSplitter:
    """Splitter with single split no training data and full test data"""

    def __init__(self):
        pass

    def split(self, X, y=None, groups=None):
        yield np.asarray([], dtype=int), np.arange(0, len(X))

    def get_n_splits(self, X, y=None, groups=None):
        return 1


class FuzzyGridTune:
    """An exhaustive FIS tuner"""

    def __init__(self, fis, params, scoring="neg_root_mean_squared_error", n_jobs=None):
        """

        Args:
            fis (FIS): The Fuzzy Inference System to tune
            params (dict of str to list): A mapping from encoded parameters to the list of values to explore.
            scoring (str): The metric used for scoring. Must be one of sklearn's regression scorings.
            n_jobs (int): Number of jobs to run in parallel.

        """
        # Grid parameter tuning
        if GridSearchCV is None:
            raise ModuleNotFoundError("scikit-learn is required for model tuning")
        self.fis = fis
        self.cv = GridSearchCV(ParametrizedFIS(fis),
                               params,
                               scoring=scoring,
                               cv=TrivialSplitter(),
                               refit=False,
                               n_jobs=n_jobs
                               )

    def fit(self, X, y=None):
        """

        Args:
            X (2D array-like): An object suitable for FIS.batch_predict.
            y (1D array-like): An array with true values. If None, but X is a DataFrame, the values are extracted from
                               there.

        Returns:

        """
        # Try to extract y if a single dataframe is provided
        if y is None and pd is not None and isinstance(X, pd.DataFrame):
            y = X[self.fis.target.name]

        self.cv.fit(X, y)

        self.best_params_ = self.cv.best_params_
        self.results = self.cv.cv_results_
        self.tuned_fis_ = _set_vars(self.fis, self.cv.best_params_)
