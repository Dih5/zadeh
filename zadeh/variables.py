try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from .domains import Domain
from .sets import FuzzySet
from .rules import FuzzyValuation, FuzzyNotValuation


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

    def plot(self):
        """Plot the membership function for each of the values"""
        for value, set in self.values.items():
            self.domain.plot_set(set, label=value)

        plt.legend()

    def __getitem__(self, item):
        return self.values[item]
