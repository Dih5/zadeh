from math import exp


class FuzzySet:
    """A fuzzy set"""

    def __init__(self, mu=None):
        self.mu = mu

    def get_description(self):
        raise NotImplementedError("Descriptions are only available for primitive types")

    @staticmethod
    def from_description(description):
        return _set_types[description["type"]].from_description(description)

    def __call__(self, x):
        assert self.mu is not None, "A membership function has to be defined"
        return self.mu(x)

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

    def get_description(self):
        return {"type": "singleton", "x": self.x}

    @staticmethod
    def from_description(description):
        return SingletonSet(description["x"])


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

    def get_description(self):
        return {"type": "discrete", "d": self.d}

    @staticmethod
    def from_description(description):
        return DiscreteFuzzySet(description["d"])


class TriangularFuzzySet(FuzzySet):
    """A fuzzy set defined by a triangular function"""

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        super().__init__()

    def get_description(self):
        return {"type": "triangular", "a": self.a, "b": self.b, "c": self.c}

    @staticmethod
    def from_description(description):
        return TriangularFuzzySet(description["a"], description["b"], description["c"])

    def __call__(self, x):
        return max(min((x - self.a) / (self.b - self.a), (self.c - x) / (self.c - self.b)), 0)


class TrapezoidalFuzzySet(FuzzySet):
    """A fuzzy set defined by a trapezoidal function"""

    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        super().__init__()

    def get_description(self):
        return {"type": "trapezoidal", "a": self.a, "b": self.b, "c": self.c, "d": self.d}

    @staticmethod
    def from_description(description):
        return TrapezoidalFuzzySet(description["a"], description["b"], description["c"], description["d"])

    def __call__(self, x):
        return max(min((x - self.a) / (self.b - self.a), 1, (self.d - x) / (self.d - self.c)), 0)


class GaussianFuzzySet(FuzzySet):
    """A fuzzy set defined by a gaussian function"""

    def __init__(self, a, c):
        self.a = a
        self.c = c
        super().__init__()

    def get_description(self):
        return {"type": "gaussian", "a": self.a, "c": self.c}

    @staticmethod
    def from_description(description):
        return GaussianFuzzySet(description["a"], description["c"])

    def __call__(self, x):
        return exp(-((x - self.c) / self.a) ** 2 / 2)


class BellFuzzySet(FuzzySet):
    """A fuzzy set defined by a generalized Bell MF

    :math:`\\mu_{a,b,c}(x)= \\frac{1}{1+\\left|\frac{x-c}{a}\\right|^{2b}}`
    """

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        super().__init__()

    def get_description(self):
        return {"type": "bell", "a": self.a, "b": self.b, "c": self.c}

    @staticmethod
    def from_description(description):
        return BellFuzzySet(description["a"], description["b"], description["c"])

    def __call__(self, x):
        return 1 / (1 + ((x - self.c) / self.a) ** (2 * self.b))


class SigmoidalFuzzySet(FuzzySet):
    """A fuzzy set defined by a sigmoid function"""

    def __init__(self, a, c):
        self.a = a
        self.c = c
        super().__init__()

    def get_description(self):
        return {"type": "sigmoidal", "a": self.a, "c": self.c}

    @staticmethod
    def from_description(description):
        return SigmoidalFuzzySet(description["a"], description["c"])

    def __call__(self, x):
        return 1 / (1 + exp(-self.a * (x - self.c)))


_set_types = {"singleton": SingletonSet, "discrete": DiscreteFuzzySet, "sigmoid": SigmoidalFuzzySet,
              "bell": BellFuzzySet, "gaussian": GaussianFuzzySet, "trapezoidal": TrapezoidalFuzzySet,
              "triangular": TriangularFuzzySet}
