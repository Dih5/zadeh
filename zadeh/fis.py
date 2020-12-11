import json
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import ipywidgets
except ImportError:
    ipywidgets = None

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Activates 3d perspective
    from matplotlib import cm
except ImportError:
    plt = None

from .variables import FuzzyVariable
from .rules import FuzzyRuleSet
from .context import FuzzyContext, set_fuzzy_context


class FIS:
    """A fuzzy inference system"""

    def __init__(self, variables, rules, target, defuzzification="centroid", aggregation="max", implication="min",
                 AND="min", OR="max"):
        self.variables = variables
        if not isinstance(rules, FuzzyRuleSet):
            rules = FuzzyRuleSet(rules)
        self.rules = rules

        # TODO: Support multitarget
        self.target = target

        self.context = FuzzyContext(defuzzification=defuzzification, aggregation=aggregation, implication=implication,
                                    AND=AND, OR=OR)



    def save(self, path):
        """Save the FIS definition to a path"""
        with open(path, "w") as f:
            json.dump(self._get_description(), f)

    @staticmethod
    def load(path):
        """Load a FIS from the given path"""
        with open(path) as f:
            s = json.load(f)
        return FIS._from_description(s)

    def _get_description(self):
        return {"variables": [v._get_description() for v in self.variables],
                "rules": self.rules._get_description(),
                "target": self.target._get_description(),
                "defuzzification": self.context.defuzzification,
                "aggregation": self.context.aggregation,
                "implication": self.context.implication,
                "AND": self.context.AND,
                "OR": self.context.OR,
                }

    @staticmethod
    def _from_description(description):
        variables = [FuzzyVariable._from_description(d) for d in description["variables"]]
        target_variable = FuzzyVariable._from_description(description["target"])
        variables_dict = {**{v.name: v for v in variables}, target_variable.name: target_variable}

        defuzzification = description.get("defuzzification", "centroid")
        aggregation = description.get("aggregation", "max")
        implication = description.get("implication", "min")
        OR = description.get("OR", "max")
        AND = description.get("AND", "min")
        return FIS(variables,
                   FuzzyRuleSet._from_description(description["rules"], variables_dict),
                   target_variable,
                   defuzzification=defuzzification, aggregation=aggregation, implication=implication,
                   AND=AND, OR=OR)

    def _to_c(self):
        with set_fuzzy_context(self.context):
            return self.rules._to_c()

    def get_output(self, values):
        """
        Get the output of the system as a fuzzy set

        Args:
            values (dict of str): A mapping from variables to their fuzzy values.

        Returns:
            FuzzySet: The fuzzy set

        """
        with set_fuzzy_context(self.context):
            return self.rules(values)

    def get_crisp_output(self, values):
        """
        Get the output of the system as a crisp value

        Args:
            values (dict of str): A mapping from variables to their fuzzy values.

        Returns:
            Centroid of the output of the system

        """
        with set_fuzzy_context(self.context):
            return self.target.domain.defuzzify(self.get_output(values))

    def batch_predict(self, X):
        """
        Get the crisp output for a batch of inputs

        Args:
            X (pd.DataFrame or np.array or list of list): Input values. If pandas dataframe, must have a column for
                                                          each of the variables with the same name. If array-like, order
                                                          must be consistent with the variables.

        Returns:
            np.array: An array with the predictions.

        """
        # Pandas dataframe syntax -- if available
        if pd is not None and isinstance(X, pd.DataFrame):
            return np.asarray(
                [self.get_crisp_output({v.name: x[v.name] for v in self.variables}) for _, x in X.iterrows()])

        # Assuming ordered array-like input
        return np.asarray([self.get_crisp_output({v.name: x[i] for i, v in enumerate(self.variables)}) for x in X])

    def dict_to_ordered(self, values):
        """Transform a dict of inputs into an array in the FIS order"""
        return [values[key.name] for key in self.variables]

    def get_interactive(self, continuous_update=False):
        """
        Display an interactive plot with the fuzzy output of the FIS

        Args:
            continuous_update (bool): Whether to continuously update with the widgets value.

        """
        if ipywidgets is None or plt is None:
            raise ModuleNotFoundError("ipywidgets and matplotlib are required")

        def plot(**kwargs):
            output = self.get_output(kwargs)
            with set_fuzzy_context(self.context):
                crisp_value = self.target.domain.defuzzify(output)
            self.target.domain.plot_set(output)
            plt.vlines(crisp_value, *plt.ylim(), color="red")
            plt.legend(["Fuzzy output", "Crisp"])
            plt.show()

        ipywidgets.interact(plot,
                            **{variable.name: variable.domain.get_ipywidget(continuous_update=continuous_update) for
                               variable in self.variables})

    def plot_1d(self, variable, fixed_variables=None, axes=None):
        """
        Produce a plot with the output as a function of a variable when the rest are fixed.

        Args:
            variable (FuzzyVariable): The independent variable.
            fixed_variables (dict of str): A mapping with fuzzy values of the rest of the variables.
            axes (plt.Axes): An existing axes instance to plot. If None, a new figure is created.

        Returns:
            plt.Axes: Axes for further tweaking

        """
        if plt is None:
            raise ModuleNotFoundError("matplotlib is required")

        if fixed_variables is None:
            fixed_variables = {}

        xx = variable.domain.get_mesh()
        output = [self.get_crisp_output({variable.name: x, **fixed_variables}) for x in xx]

        ax = axes or plt.figure().add_subplot(1, 1, 1)
        ax.plot(xx, output)
        ax.set_xlabel(variable.name)
        ax.set_ylabel(self.target.name)
        return ax

    def get_1d_interactive(self, variable, continuous_update=False):
        """
        Produce an interactive plot with the output as a function of a variable when the rest are fixed.

        Args:
            variable (FuzzyVariable): The independent variable.
            continuous_update (bool): Whether to continuously update with the widgets value.

        """
        if ipywidgets is None or plt is None:
            raise ModuleNotFoundError("ipywidgets and matplotlib are required")

        free_variables = [v for v in self.variables if v.name != variable.name]

        def plot(**kwargs):
            self.plot_1d(variable, kwargs)
            plt.show()

        ipywidgets.interact(plot,
                            **{variable.name: variable.domain.get_ipywidget(continuous_update=continuous_update) for
                               variable in free_variables})

    def plot_2d(self, variable1, variable2, fixed_variables=None, axes=None):
        """
        Produce a plot with the output as a function of two variables when the rest are fixed.

        Args:
            variable1 (FuzzyVariable): The first independent variable.
            variable2 (FuzzyVariable): The second independent variable.
            fixed_variables (dict of str): A mapping with fuzzy values of the rest of the variables.
            axes (plt.Axes): An existing axes instance to plot. An 3D projection must have been set on it. If None,
                             a new figure is created.

        Returns:
            plt.Axes: Axes for further tweaking

        """
        if plt is None:
            raise ModuleNotFoundError("matplotlib is required")

        if fixed_variables is None:
            fixed_variables = {}

        x_name = variable1.name
        y_name = variable2.name

        # TODO: Allow coarser mesh
        xx = variable1.domain.get_mesh()
        yy = variable2.domain.get_mesh()

        zz = np.asarray(
            [
                [
                    self.get_crisp_output({x_name: x, y_name: y, **fixed_variables})
                    for x in xx
                ]
                for y in yy
            ]
        )

        # String coordinates must be converted for this kind of plot:
        if xx.dtype.kind == 'U':
            xx = np.arange(len(xx))
        if yy.dtype.kind == 'U':
            yy = np.arange(len(yy))

        ax = axes or plt.figure().add_subplot(1, 1, 1, projection="3d")
        ax.plot_surface(*np.meshgrid(xx, yy), zz, cmap=cm.viridis)

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel(self.target.name)

        ax.invert_xaxis()  # Seems more natural to me

        return ax

    def get_2d_interactive(self, variable1, variable2, continuous_update=False):
        """
        Produce an interactive plot with the output as a function of two variables when the rest are fixed.

        Args:
            variable1 (FuzzyVariable): The first independent variable.
            variable2 (FuzzyVariable): The second independent variable.
            continuous_update (bool): Whether to continuously update with the widgets value.

        """
        if ipywidgets is None or plt is None:
            raise ModuleNotFoundError("ipywidgets and matplotlib are required")

        free_variables = [v for v in self.variables if (v.name != variable1.name and v.name != variable2.name)]

        def plot(**kwargs):
            self.plot_2d(variable1, variable2, kwargs)
            plt.show()

        ipywidgets.interact(plot,
                            **{variable.name: variable.domain.get_ipywidget(continuous_update=continuous_update) for
                               variable in free_variables})

    def compile(self):
        """Get a compiled version of the model"""
        from .compile import CompiledFIS
        return CompiledFIS.from_existing(self)

    @staticmethod
    def from_matlab(path):
        """Import a MATLAB® model from a .fis file"""
        from .mparser import read_mfis
        return read_mfis(path)
