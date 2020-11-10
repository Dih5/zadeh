try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from .domains import Domain
from .sets import FuzzySet


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

    def get_description(self):
        return {"name": self.name,
                "values": {key: val.get_description() for key, val in self.values.items()},
                "domain": self.domain.get_description()}

    @staticmethod
    def from_description(description):
        domain = Domain.from_description(description["domain"])
        values = {key: FuzzySet.from_description(val) for key, val in description["values"].items()}
        return FuzzyVariable(domain, values, name=description["name"])

    def plot(self):
        """Plot the membership function for each of the values"""
        for value, set in self.values.items():
            self.domain.plot_set(set, label=value)

        plt.legend()

    def __getitem__(self, item):
        return self.values[item]
