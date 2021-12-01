try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import numpy as np

from .domains import Domain, FloatDomain
from .sets import FuzzySet
from .rules import FuzzyValuation, FuzzyNotValuation
from . import sets


class FuzzyVariable:
    """A fuzzy variable"""

    def __init__(self, domain, values, name=None):
        """

        Args:
            domain (Domain): Domain (universe of discourse) of the variable.
            values (dict of str to FuzzySet): Mapping of values to fuzzy numbers.
            name (str): The name of the variable. If none, taken from the domain name.
        """
        self.domain = domain
        self.values = values
        self.name = name if name is not None else self.domain.name

    @staticmethod
    def automatic(name, min, max, steps, values, endpoints=True, value_names=None, width_factor=1.0, shape="gaussian"):
        """
        Automatically create a fuzzy variable with the number of values provided.

        Args:
            name (str): Name of the variable and of its domain.
            min (float): Minimum value of the domain
            max (float): Maximum value of the domain
            steps (int or float): Number of steps if int or step size if float.
            values (int): Number of values.
            endpoints (bool): Whether the start and ending points are considered.
            value_names (list of str): Names of values. If not provided or if length does not match, they will be
                                       automatically guessed when possible.
            width_factor (float): A scale factor for the width of the membership function defining the values.
            shape (str): Kind of sets used to define the values. Available options are:
                         - gaussian: Gaussian sets.
                         - triangular: Triangular sets.
                         - trapezoidal: Trapezoidal sets.
                         - spline: order-2 spline-based sets (S, Pi, Z).
                         - sigmoidald: Sigmoidal functions, using the sigmoidal difference for non-endpoints.
                         - sigmoidalp: Sigmoidal functions, using the sigmoidal product for non-endpoints.

        Returns:
            FuzzyVariable: The automatically generated fuzzy variable.

        """
        return _auto_variable(name,
                              min,
                              max,
                              steps,
                              values,
                              endpoints=endpoints,
                              value_names=value_names,
                              width_factor=width_factor,
                              shape=shape,
                              )

    def __eq__(self, other):
        if not isinstance(other, str):
            raise ValueError("FuzzyVariable can only be compared to str values")
        return FuzzyValuation(self, other)

    def __ne__(self, other):
        if not isinstance(other, str):
            raise ValueError("FuzzyVariable can only be compared to str values")
        return FuzzyNotValuation(self, other)

    def _get_description(self):
        return {"name": self.name,
                "values": {key: val._get_description() for key, val in self.values.items()},
                "domain": self.domain._get_description()}

    @staticmethod
    def _from_description(description):
        domain = Domain._from_description(description["domain"])
        values = {key: FuzzySet._from_description(val) for key, val in description["values"].items()}
        return FuzzyVariable(domain, values, name=description["name"])

    def plot(self, value=None):
        """
        Plot the membership function for each of the values

        Args:
            value (str): A value to highlight. If so, the other values are shown dimmed and not in the legend

        """
        for val, set in self.values.items():
            if value is not None:
                if val == value:
                    self.domain.plot_set(set, label=val)
                else:
                    self.domain.plot_set(set, alpha=0.3)
            else:  # Plot all values
                self.domain.plot_set(set, label=val)

        plt.legend()

    def __getitem__(self, item):
        return self.values[item]


# Automatic fuzzy value generation

# Dictionaries for shapes which are different in their endpoints
_spline_converters = {-1: lambda x, w: sets.ZFuzzySet(x, x + w),
                      0: lambda x, w: sets.PiFuzzySet(x - w * 0.75, x - w * 0.25, x + w * 0.25, x + w * 0.75),
                      1: lambda x, w: sets.SFuzzySet(x - w, x)}

_sigmoidald_converters = {-1: lambda x, w: sets.SigmoidalFuzzySet(-1 / w * 8, x + w / 2),
                          0: lambda x, w: sets.SigmoidalDifferenceFuzzySet(1 / w * 8, x - w / 2, 1 / w * 8, x + w / 2),
                          1: lambda x, w: sets.SigmoidalFuzzySet(1 / w * 8, x - w / 2)}

_sigmoidalp_converters = {-1: lambda x, w: sets.SigmoidalFuzzySet(-1 / w * 8, x + w / 2),
                          0: lambda x, w: sets.SigmoidalProductFuzzySet(1 / w * 8, x - w / 2, -1 / w * 8, x + w / 2),
                          1: lambda x, w: sets.SigmoidalFuzzySet(1 / w * 8, x - w / 2)}

# Converter functions, mapping a position, a width, and a label identifying start/mid/end to the suitable arguments
# The width meaning is class-dependent, but default scaling tries to provide consistent values
_converters = {"gaussian": lambda x, w, i: sets.GaussianFuzzySet(w * 0.5 / 1.414, x),  # scaled by sqrt(2)
               "triangular": lambda x, w, i: sets.TriangularFuzzySet(x - w * 0.75, x, x + w * 0.75),
               # Note there are actually two scales in trapezoidal
               "trapezoidal": lambda x, w, i: sets.TrapezoidalFuzzySet(x - w * 0.75, x - w * 0.25, x + w * 0.25,
                                                                       x + w * 0.75),
               "spline": lambda x, w, i: _spline_converters[i](x, w),
               "sigmoidald": lambda x, w, i: _sigmoidald_converters[i](x, w),
               "sigmoidalp": lambda x, w, i: _sigmoidalp_converters[i](x, w),
               }


def _auto_variable(name, min, max, steps, values, endpoints=True, value_names=None, width_factor=1.0, shape="gaussian"):
    if endpoints:
        xx = np.linspace(min, max, values, endpoint=True)
    else:
        xx = np.linspace(min, max, values + 1, endpoint=False)[1:]

    step = xx[1] - xx[0]

    if value_names is None:
        if values <= 7:
            # Will use semantic description
            # For values in [4, 7], see below
            if values % 2:
                value_names = ["low", "medium", "high"]
            else:
                value_names = ["low", "high"]
        else:
            value_names = ["val%d" % d for d in range(1, values + 1)]

    # Some automatic extension mechanisms
    if len(value_names) == 2 and values == 4 or len(value_names) == 3 and values == 5:
        value_names = (
            ["very " + value_names[0]] + value_names + ["very " + value_names[-1]]
        )
    if len(value_names) == 2 and values == 6 or len(value_names) == 3 and values == 7:
        value_names = (
            ["very very " + value_names[0], "very " + value_names[0]]
            + value_names
            + ["very " + value_names[-1], "very very " + value_names[-1]]
        )

    if endpoints:
        kinds = [-1] + [0] * (values - 2) + [1]
    else:
        kinds = [0] * values

    f = _converters[shape.lower()]
    _values = {name: f(x, step * width_factor, kind) for name, x, kind in zip(value_names, xx, kinds)}

    return FuzzyVariable(FloatDomain(name, min, max, steps), _values)
