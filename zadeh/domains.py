import numpy as np

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

    @staticmethod
    def from_description(description):
        return _domain_subclasses[description["type"]].from_description(description)

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

    def centroid(self, set):
        """Calculate the centroid of a fuzzy set"""
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

    def get_description(self):
        return {"type": "FloatDomain", "name": self.name, "min": self.min, "max": self.max, "steps": self.steps}

    @staticmethod
    def from_description(description):
        return FloatDomain(description["name"], description["min"], description["max"], description["steps"])

    def get_mesh(self):
        if self.steps is None or isinstance(self.steps, int):
            return np.linspace(self.min, self.max, self.steps)
        elif isinstance(self.steps, float):
            return np.arange(self.min, self.max, self.steps)
        else:
            raise ValueError("Bad type for steps")

    def centroid(self, set):
        xx, mu = self.evaluate_set(set)
        return np.average(xx, weights=mu)

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

    def get_description(self):
        return {"type": "CategoricalDomain", "name": self.name, "values": self.values}

    @staticmethod
    def from_description(description):
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
