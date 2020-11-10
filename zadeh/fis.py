import numpy as np

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


class FIS:
    """A fuzzy inference system"""

    def __init__(self, variables, rules, target):
        self.variables = variables
        self.rules = rules

        # TODO: Support multitarget
        self.target = target

    def get_description(self):
        return {"variables": [v.get_description() for v in self.variables],
                "rules": self.rules.get_description(),
                "target": self.target.get_description()}

    @staticmethod
    def from_description(description):
        variables = [FuzzyVariable.from_description(d) for d in description["variables"]]
        target_variable = FuzzyVariable.from_description(description["target"])
        variables_dict = {**{v.name: v for v in variables}, target_variable.name: target_variable}
        return FIS(variables,
                   FuzzyRuleSet.from_description(description["rules"], variables_dict),
                   target_variable)

    def get_output(self, values):
        return self.rules(values)

    def get_crisp_output(self, values):
        return self.target.domain.centroid(self.get_output(values))

    def get_interactive(self):
        if ipywidgets is None or plt is None:
            raise ModuleNotFoundError("ipywidgets and matplotlib are required")

        def plot(**kwargs):
            output = self.get_output(kwargs)
            self.target.domain.plot_set(output)
            plt.vlines(self.target.domain.centroid(output), *plt.ylim(), color="red")
            plt.legend(["Fuzzy output", "Centroid"])
            plt.show()

        ipywidgets.interact(plot, **{variable.name: variable.domain.get_ipywidget() for variable in self.variables})

    def plot_surface(self):
        if len(self.variables) < 2:
            raise ValueError("At least two variables are required")
        if len(self.variables) > 2:
            # TODO: Allow selection of additional variables as fixed values
            raise NotImplementedError

        x_name = self.variables[0].name
        y_name = self.variables[1].name
        # TODO: Allow coarser mesh
        xx = self.variables[0].domain.get_mesh()
        yy = self.variables[1].domain.get_mesh()
        zz = np.asarray(
            [
                [
                    self.get_crisp_output({x_name: x, y_name: y})
                    for x in xx
                ]
                for y in yy
            ]
        )

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(*np.meshgrid(xx, yy), zz, cmap=cm.viridis)

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel(self.target.name)

        ax.invert_xaxis()  # Seems more natural to me

        return ax
