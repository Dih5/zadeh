from math import exp

import numpy as np


def _ensure_spaced(m, eps=1E-8):
    """Ensure a list of numbers is ordered and with no repetitions, slightly shifting them if needed"""
    m = np.asarray(m)
    d = np.diff(m)
    if np.any(d < 0):
        raise ValueError("List of numbers is not ordered")
    return np.cumsum(np.concatenate(([m[0]], np.where(d > 0, d, eps))))


class FuzzySet:
    """A fuzzy set"""

    def __init__(self, mu=None):
        self.mu = mu

    def _get_description(self):
        raise NotImplementedError("Descriptions are only available for primitive types")

    @staticmethod
    def _from_description(description):
        return _set_types[description["type"]]._from_description(description)

    def __call__(self, x):
        assert self.mu is not None, "A membership function has to be defined"
        return self.mu(x)

    def _to_c(self, name):
        raise NotImplementedError("C code generation not available. Overwrite the _to_c method if you know what "
                                  "you are doing.")

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return FuzzySet(lambda x: other * self(x))
        raise NotImplementedError("Multiplication is only defined with crisp numbers")

    def __rmul__(self, other):
        return self.__mul__(other)

    # TODO: More generalized fuzzy set operations could be defined

    def __neg__(self):
        return FuzzySet(lambda x: 1 - self(x))

    def __or__(self, other):
        return FuzzySet(lambda x: max(self(x), other(x)))

    def __and__(self, other):
        return FuzzySet(lambda x: min(self(x), other(x)))

    @staticmethod
    def n_ary_or(sets):
        return FuzzySet(lambda x: max(s(x) for s in sets))

    @staticmethod
    def n_ary_and(sets):
        return FuzzySet(lambda x: min(s(x) for s in sets))


class SingletonSet(FuzzySet):
    """A singleton fuzzy set (Kronecker delta)"""

    def __init__(self, x):
        """

        Args:
            x: The unique value where membership is 1.
        """
        super().__init__(lambda y: 1 if x == y else 0)
        self.x = x

    def _get_description(self):
        return {"type": "singleton", "x": self.x}

    @staticmethod
    def _from_description(description):
        return SingletonSet(description["x"])

    def _to_c(self, name):
        return "({x}=={a})?1.0:0.0".format(x=name, a=self.x)


class DiscreteFuzzySet(FuzzySet):
    """A discrete fuzzy set (non-null in a discrete set of points)"""

    def __init__(self, d):
        """

        Args:
            d (dict): Mapping of values to non-null membership values

        """
        self.d = d
        super().__init__()

    def __call__(self, x):
        return self.d.get(x, 0)

    def _to_c(self, name):
        raise NotImplementedError("C code not available for DiscreteFuzzySet")

    def _get_description(self):
        return {"type": "discrete", "d": self.d}

    @staticmethod
    def _from_description(description):
        return DiscreteFuzzySet(description["d"])


class TriangularFuzzySet(FuzzySet):
    """A fuzzy set defined by a triangular function"""

    def __init__(self, a, b, c):
        a, b, c = _ensure_spaced([a, b, c])
        self.a = a
        self.b = b
        self.c = c
        super().__init__()

    def _get_description(self):
        return {"type": "triangular", "a": self.a, "b": self.b, "c": self.c}

    @staticmethod
    def _from_description(description):
        return TriangularFuzzySet(description["a"], description["b"], description["c"])

    def __call__(self, x):
        return max(min((x - self.a) / (self.b - self.a), (self.c - x) / (self.c - self.b)), 0)

    def _to_c(self, name):
        return "max(2, min(2, ({x} - {a}) / ({b} - {a}), ({c} - {x}) / ({c} - {b})), 0.0)".format(x=name,
                                                                                                  a=self.a,
                                                                                                  b=self.b,
                                                                                                  c=self.c)


class TrapezoidalFuzzySet(FuzzySet):
    """A fuzzy set defined by a trapezoidal function"""

    def __init__(self, a, b, c, d):
        a, b, c, d = _ensure_spaced([a, b, c, d])
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        super().__init__()

    def _get_description(self):
        return {"type": "trapezoidal", "a": self.a, "b": self.b, "c": self.c, "d": self.d}

    @staticmethod
    def _from_description(description):
        return TrapezoidalFuzzySet(description["a"], description["b"], description["c"], description["d"])

    def __call__(self, x):
        return max(min((x - self.a) / (self.b - self.a), 1, (self.d - x) / (self.d - self.c)), 0)

    def _to_c(self, name):
        return "max(2, min(3, ({x} - {a}) / ({b} - {a}), 1.0, ({d} - {x}) / ({d} - {c})), 0.0)".format(x=name,
                                                                                                       a=self.a,
                                                                                                       b=self.b,
                                                                                                       c=self.c,
                                                                                                       d=self.d)


class GaussianFuzzySet(FuzzySet):
    """A fuzzy set defined by a gaussian function"""

    def __init__(self, a, c):
        self.a = a
        self.c = c
        super().__init__()

    def _get_description(self):
        return {"type": "gaussian", "a": self.a, "c": self.c}

    @staticmethod
    def _from_description(description):
        return GaussianFuzzySet(description["a"], description["c"])

    def __call__(self, x):
        return exp(-((x - self.c) / self.a) ** 2 / 2)

    def _to_c(self, name):
        return "exp(- pow(({x}-{c})/{a}, 2.0) / 2.0)".format(x=name, a=self.a, c=self.c)


class BellFuzzySet(FuzzySet):
    """A fuzzy set defined by a generalized Bell MF

    :math:`\\mu_{a,b,c}(x)= \\frac{1}{1+\\left|\frac{x-c}{a}\\right|^{2b}}`
    """

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        super().__init__()

    def _get_description(self):
        return {"type": "bell", "a": self.a, "b": self.b, "c": self.c}

    @staticmethod
    def _from_description(description):
        return BellFuzzySet(description["a"], description["b"], description["c"])

    def __call__(self, x):
        return 1 / (1 + ((x - self.c) / self.a) ** (2 * self.b))

    def _to_c(self, name):
        return "1 / (1 + pow(({x} - {c}) / {a}, 2*{b}) )".format(x=name, a=self.a, b=self.b, c=self.c)


class SigmoidalFuzzySet(FuzzySet):
    """A fuzzy set defined by a sigmoid function"""

    def __init__(self, a, c):
        self.a = a
        self.c = c
        super().__init__()

    def _get_description(self):
        return {"type": "sigmoidal", "a": self.a, "c": self.c}

    @staticmethod
    def _from_description(description):
        return SigmoidalFuzzySet(description["a"], description["c"])

    def __call__(self, x):
        return 1 / (1 + exp(-self.a * (x - self.c)))

    def _to_c(self, name):
        return "1 / (1 + exp(-{a} * ({x} - {c})))".format(x=name, a=self.a, c=self.c)


_set_types = {"singleton": SingletonSet, "discrete": DiscreteFuzzySet, "sigmoid": SigmoidalFuzzySet,
              "bell": BellFuzzySet, "gaussian": GaussianFuzzySet, "trapezoidal": TrapezoidalFuzzySet,
              "triangular": TriangularFuzzySet}
