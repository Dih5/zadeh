import bisect
import numpy as np

from .context import get_active_context

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import ipywidgets
except ImportError:
    ipywidgets = None


class Domain:
    """A domain where diffuse sets are defined"""

    def __init__(self, name):
        self.name = name

    def _get_description(self):
        raise NotImplementedError

    @staticmethod
    def _from_description(description):
        return _domain_subclasses[description["type"]]._from_description(description)

    def get_mesh(self):
        """Get a mesh representing the domain"""
        raise NotImplementedError("A domain subclass must be used instead.")

    def evaluate_set(self, set):
        """Get a pair of list with a mesh representing the domain and the evaluation of a membership function on it"""
        mesh = self.get_mesh()
        return mesh, [set(x) for x in mesh]

    def plot_set(self, set, **kwargs):
        """Plot a fuzzy set"""
        if plt is None:
            raise ModuleNotFoundError("Matplotlib is required for plotting")
        plt.plot(*self.evaluate_set(set), **kwargs)
        plt.xlabel(self.name)
        plt.ylabel("Membership function")

    def defuzzify(self, set):
        """Calculate a crisp number from the fuzzy set"""
        raise NotImplementedError("A Domain subclass must be used instead.")

    def get_ipywidget(self, **kwargs):
        """Get a widget representing the domain"""
        raise NotImplementedError("A Domain subclass must be used instead.")


class FloatDomain(Domain):
    def __init__(self, name, min, max, steps):
        """

        Args:
            name (str): Name of the domain.
            min (float): Minimum value of the domain
            max (float): Maximum value of the domain
            steps (int or float): Number of steps if int or step size if float.
        """
        super().__init__(name)
        self.min = min
        self.max = max
        self.steps = steps

    def _get_description(self):
        return {"type": "FloatDomain", "name": self.name, "min": self.min, "max": self.max, "steps": self.steps}

    @staticmethod
    def _from_description(description):
        return FloatDomain(description["name"], description["min"], description["max"], description["steps"])

    def get_mesh(self):
        if self.steps is None or isinstance(self.steps, int):
            return np.linspace(self.min, self.max, self.steps)
        elif isinstance(self.steps, float):
            return np.arange(self.min, self.max, self.steps)
        else:
            raise ValueError("Bad type for steps")

    def defuzzify(self, set):
        method = get_active_context().defuzzification
        if method == "centroid":
            return self.centroid(set)
        elif method == "bisector":
            return self.bisector(set)
        elif method == "mom":
            return self.mom(set)
        elif method == "som":
            return self.som(set)
        elif method == "lom":
            return self.lom(set)

        raise ValueError("Invalid defuzzification method in context: %s" % method)

    def centroid(self, set):
        """Defuzzify with the centroid (center of mass)"""
        xx, mu = self.evaluate_set(set)
        try:
            return np.average(xx, weights=mu)
        except ZeroDivisionError:
            return np.nan

    def bisector(self, set):
        """Defuzzify with the bisector (value separating two portions of equal area under the membership function)"""
        xx, mu = self.evaluate_set(set)

        cum_mu = np.cumsum(mu)
        # TODO: This could actually be improved interpolating with the nearest values
        mean_pos = bisect.bisect_left(cum_mu, cum_mu[-1] / 2)

        return xx[mean_pos]

    def mom(self, set):
        """Defuzzify with the middle of maximum"""
        xx, mu = self.evaluate_set(set)
        max_mu = max(mu)
        xx_max = [x for x, m in zip(xx, mu) if m == max_mu]
        return np.median(xx_max)

    def som(self, set):
        """Defuzzify with the smaller of maximum"""
        xx, mu = self.evaluate_set(set)
        max_mu = max(mu)
        xx_max = [x for x, m in zip(xx, mu) if m == max_mu]
        return min(xx_max)

    def lom(self, set):
        """Defuzzify with the largest of maximum"""
        xx, mu = self.evaluate_set(set)
        max_mu = max(mu)
        xx_max = [x for x, m in zip(xx, mu) if m == max_mu]
        return max(xx_max)

    def get_ipywidget(self, **kwargs):
        if ipywidgets is None:
            raise ModuleNotFoundError("ipywidgets is required")
        kwargs = {k: v for k, v in kwargs.items() if k in ["continuous_update"]}
        return ipywidgets.FloatSlider(min=self.min, max=self.max, **kwargs)


class CategoricalDomain(Domain):
    def __init__(self, name, values):
        """

        Args:
            name (str): Name of the domain.
            values (list): List of possible valued.
        """
        super().__init__(name)
        self.values = values

    def _get_description(self):
        return {"type": "CategoricalDomain", "name": self.name, "values": self.values}

    @staticmethod
    def _from_description(description):
        return CategoricalDomain(description["name"], description["values"])

    def get_mesh(self):
        return np.asarray(self.values)

    def centroid(self, set):
        # Return as the mode
        # In case of ties, only the first value is returned
        xx, mu = self.evaluate_set(set)
        return xx[np.argmax(mu)]

    def get_ipywidget(self, **kwargs):
        if ipywidgets is None:
            raise ModuleNotFoundError("ipywidgets is required")

        kwargs = {k: v for k, v in kwargs.items() if k in ["continuous_update"]}
        return ipywidgets.Dropdown(options=self.values, **kwargs)


_domain_subclasses = {"FloatDomain": FloatDomain, "CategoricalDomain": CategoricalDomain}
