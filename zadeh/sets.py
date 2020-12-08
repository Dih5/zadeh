from math import exp

import numpy as np


def _ensure_spaced(m, eps=1E-8):
    """Ensure a list of numbers is ordered and with no repetitions, slightly shifting them if needed"""
    m = np.asarray(m)
    d = np.diff(m)
    if np.any(d < 0):
        raise ValueError("List of numbers is not ordered")
    return np.cumsum(np.concatenate(([m[0]], np.where(d > 0, d, eps))))


def _clip(x, min=0, max=1):
    """Clip to interval, defaults to [0, 1]"""
    return np.clip(x, min, max)


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
            return FuzzySetScaled(self, other)
        raise NotImplementedError("Multiplication is only defined with crisp numbers")

    def __rmul__(self, other):
        return self.__mul__(other)

    # TODO: More generalized fuzzy set operations could be defined

    def __neg__(self):
        return FuzzySetNeg(self)

    def __or__(self, other):
        return FuzzySetOr([self, other])

    def __and__(self, other):
        return FuzzySetAnd([self, other])


class FuzzySetNeg(FuzzySet):
    """A negation operation on a Fuzzy set"""

    def __init__(self, set):
        super().__init__()
        self.set = set

    def __call__(self, x):
        return 1 - self.set(x)

    def _to_c(self, name):
        return "1 - (%s)" % self.set._to_c(name)


class FuzzySetOr(FuzzySet):
    """An OR operation between Fuzzy sets"""

    def __init__(self, sets):
        super().__init__()
        self.sets = sets

    def __call__(self, x):
        return max(s(x) for s in self.sets)

    def _to_c(self, name):
        return "max(%d, %s)" % (len(self.sets), ", ".join(s._to_c(name) for s in self.sets))


class FuzzySetAnd(FuzzySet):
    """An AND operation between Fuzzy sets"""

    def __init__(self, sets):
        super().__init__()
        self.sets = sets

    def __call__(self, x):
        return min(s(x) for s in self.sets)

    def _to_c(self, name):
        return "min(%d, %s)" % (len(self.sets), ", ".join(s._to_c(name) for s in self.sets))


class FuzzySetScaled(FuzzySet):
    """A scaled Fuzzy sets"""

    def __init__(self, set, scale):
        super().__init__()
        self.set = set
        self.scale = scale

    def __call__(self, x):
        return self.set(x) * self.scale

    def _to_c(self, name):
        return "%f * (%s)" % (self.scale, self.set._to_c(name))


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
    """
    A fuzzy set defined by a sigmoid function

    :math:`\\sigma_{a,c}(x)= \\frac{1}{1+e^{\\left(-a (x-c)\\right)}}`

    If a>0, the sigmoid is increasing, otherwise decreasing.
    The magnitude of "a" defines the width of the transition, "c" defines its location.

    """

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


class SigmoidalProductFuzzySet(FuzzySet):
    """
    A fuzzy set defined by the product of two sigmoid functions

    :math:`\\sigma_{a_1,c_1,a_2,c_2}^\\mathrm{p}(x)=\\sigma_{a_1,c_1}(x)\\cdot \\sigma_{a_2,c_2}(x)`

    Typical unimodal membership functions are defined setting the opposite sign for (a1, a2),
    and choosing (c1, c2) enough apart for both sigmoids reach ~1 in a common subset.


    """

    def __init__(self, a1, c1, a2, c2):
        self.a1 = a1
        self.c1 = c1
        self.a2 = a2
        self.c2 = c2
        super().__init__()

    def _get_description(self):
        return {"type": "sigmoidal_product", "a1": self.a1, "c1": self.c1, "a2": self.a2, "c2": self.c2}

    @staticmethod
    def _from_description(description):
        return SigmoidalProductFuzzySet(description["a1"], description["c1"], description["a2"], description["c2"])

    def __call__(self, x):
        return (1 / (1 + exp(-self.a1 * (x - self.c1)))) * (1 / (1 + exp(-self.a2 * (x - self.c2))))

    def _to_c(self, name):
        return "(1 / (1 + exp(-{a1} * ({x} - {c1})))) * (1 / (1 + exp(-{a2} * ({x} - {c2}))))".format(x=name,
                                                                                                      a1=self.a1,
                                                                                                      c1=self.c1,
                                                                                                      a2=self.a2,
                                                                                                      c2=self.c2)


class SigmoidalDifferenceFuzzySet(FuzzySet):
    """
    A fuzzy set defined by the difference of two sigmoid functions clipped to [0, 1]

    :math:`\\sigma_{a_1,c_1,a_2,c_2}^\\mathrm{d}(x)=\\mathrm{clip}_{0,1}(\\sigma_{a_1,c_1}(x) - \\sigma_{a_2,c_2}(x))`


    Typical unimodal membership functions are defined setting the same sign for (a1, a2),
    and choosing (c1, c2) enough apart for both sigmoids reach ~1 in a common subset.
    """

    def __init__(self, a1, c1, a2, c2):
        self.a1 = a1
        self.c1 = c1
        self.a2 = a2
        self.c2 = c2
        super().__init__()

    def _get_description(self):
        return {"type": "sigmoidal_difference", "a1": self.a1, "c1": self.c1, "a2": self.a2, "c2": self.c2}

    @staticmethod
    def _from_description(description):
        return SigmoidalDifferenceFuzzySet(description["a1"], description["c1"], description["a2"], description["c2"])

    def __call__(self, x):
        return _clip((1 / (1 + exp(-self.a1 * (x - self.c1)))) - (1 / (1 + exp(-self.a2 * (x - self.c2)))))

    def _to_c(self, name):
        return "clip((1 / (1 + exp(-{a1} * ({x} - {c1})))) - (1 / (1 + exp(-{a2} * ({x} - {c2})))))".format(x=name,
                                                                                                            a1=self.a1,
                                                                                                            c1=self.c1,
                                                                                                            a2=self.a2,
                                                                                                            c2=self.c2)


def _s_shaped(x, a, b):
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    if x <= (a + b) / 2:  # (a, (a+b)/2]
        return 2.0 * ((x - a) / (b - a)) ** 2
    # ((a+b)/2, b)
    return 1.0 - 2.0 * ((x - b) / (b - a)) ** 2


class SFuzzySet(FuzzySet):
    """
    A fuzzy set defined by an S-shaped function.

    .. math::

        S_{a,b}(x) =
             \\begin{cases}
               0, &\\quad\\text{if } x\\leq a\\\\
               2\\left(\\frac{x-a}{b-a}\\right)^2, &\\quad\\text{if } a \\leq x\\leq \\frac{a+b}{2}\\\\
               1-2\\left(\\frac{x-b}{b-a}\\right)^2, &\\quad\\text{if } \\frac{a+b}{2} \\leq x \\leq b\\\\
               1, &\\quad\\text{if } x\\geq b\\\\
             \\end{cases}



    """

    def __init__(self, a, b):
        self.a = a
        self.b = b
        super().__init__()

    def _get_description(self):
        return {"type": "s_shaped", "a": self.a, "b": self.b}

    @staticmethod
    def _from_description(description):
        return SFuzzySet(description["a"], description["b"])

    def __call__(self, x):
        return _s_shaped(x, self.a, self.b)

    def _to_c(self, name):
        return "s_shaped({x}, {a}, {b})".format(x=name, a=self.a, b=self.b)


def _z_shaped(x, a, b):
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    if x <= (a + b) / 2:  # (a, (a+b)/2]
        return 1.0 - 2.0 * ((x - a) / (b - a)) ** 2
    # ((a+b)/2, b)
    return 2.0 * ((x - b) / (b - a)) ** 2


class ZFuzzySet(FuzzySet):
    """
    A fuzzy set defined by a Z-shaped function

    .. math::

        Z_{a,b}(x) =
             \\begin{cases}
               0, &\\quad\\text{if } x\\leq a\\\\
               1-2\\left(\\frac{x-a}{b-a}\\right)^2, &\\quad\\text{if } a \\leq x\\leq \\frac{a+b}{2}\\\\
               2\\left(\\frac{x-b}{b-a}\\right)^2, &\\quad\\text{if } \\frac{a+b}{2} \\leq x \\leq b\\\\
               1, &\\quad\\text{if } x\\geq b\\\\
             \\end{cases}



    """

    def __init__(self, a, b):
        self.a = a
        self.b = b
        super().__init__()

    def _get_description(self):
        return {"type": "z_shaped", "a": self.a, "b": self.b}

    @staticmethod
    def _from_description(description):
        return ZFuzzySet(description["a"], description["b"])

    def __call__(self, x):
        return _z_shaped(x, self.a, self.b)

    def _to_c(self, name):
        return "z_shaped({x}, {a}, {b})".format(x=name, a=self.a, b=self.b)


class PiFuzzySet(FuzzySet):
    """
    A fuzzy set defined by a Pi-shaped function (a combination of S-shaped MF followed by a Z-shaped MF)

    .. math::

        Z_{a,b,c,d}(x) = S_{a,b}(x)\\cdot Z_{c,d}(x)

    """

    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        super().__init__()

    def _get_description(self):
        return {"type": "pi_shaped", "a": self.a, "b": self.b, "c": self.c, "d": self.d}

    @staticmethod
    def _from_description(description):
        return PiFuzzySet(description["a"], description["b"], description["c"], description["d"])

    def __call__(self, x):
        return _s_shaped(x, self.a, self.b) * _z_shaped(x, self.c, self.d)

    def _to_c(self, name):
        return "s_shaped({x}, {a}, {b}) * z_shaped({x}, {c}, {d})".format(x=name, a=self.a, b=self.b, c=self.c,
                                                                          d=self.d)


_set_types = {"singleton": SingletonSet, "discrete": DiscreteFuzzySet, "sigmoid": SigmoidalFuzzySet,
              "sigmoidal_product": SigmoidalProductFuzzySet, "sigmoidal_difference": SigmoidalDifferenceFuzzySet,
              "s_shaped": SFuzzySet, "z_shaped": ZFuzzySet, "pi_shaped": PiFuzzySet,
              "bell": BellFuzzySet, "gaussian": GaussianFuzzySet, "trapezoidal": TrapezoidalFuzzySet,
              "triangular": TriangularFuzzySet}
