from math import exp

import numpy as np

from .context import get_active_context

try:
    from math import prod  # Python >= 3.8
except ImportError:
    def prod(xx):
        result = 1
        for x in xx:
            result *= x
        return result


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

    def __init__(self, sets, method=None):
        super().__init__()
        self.sets = sets
        self.method = method

    def __call__(self, x):
        method = get_active_context().OR if self.method is None else self.method
        if method == "max":
            return max(s(x) for s in self.sets)
        elif method == "psum":
            return 1 - prod(1 - s(x) for s in self.sets)
        elif method == "bsum":
            return min(1, sum(s(x) for s in self.sets))
        else:
            raise ValueError("Invalid OR method in context: %s" % method)

    def _to_c(self, name):
        method = get_active_context().OR if self.method is None else self.method
        if method == "max":
            return "max(%d, %s)" % (len(self.sets), ", ".join(s._to_c(name) for s in self.sets))
        elif method == "psum":
            return "1 - %s" % " * ".join("(1 - %s)" % s._to_c(name) for s in self.sets)
        elif method == "bsum":
            return "min(2, 1, %s)" % " + ".join("(%s)" % s._to_c(name) for s in self.sets)
        else:
            raise ValueError("Invalid OR method in context: %s" % method)


class FuzzySetAnd(FuzzySet):
    """An AND operation between Fuzzy sets"""

    def __init__(self, sets, method=None):
        super().__init__()
        self.sets = sets
        self.method = method

    def __call__(self, x):
        method = get_active_context().AND if self.method is None else self.method
        if method == "min":
            return min(s(x) for s in self.sets)
        elif method == "product":
            return prod(s(x) for s in self.sets)
        elif method == "lukasiewicz":
            return max(0, sum(s(x) for s in self.sets) - (len(self.sets) - 1))
        else:
            raise ValueError("Invalid AND method in context: %s" % method)

    def _to_c(self, name):
        method = get_active_context().AND if self.method is None else self.method
        if method == "min":
            return "min(%d, %s)" % (len(self.sets), ", ".join(s._to_c(name) for s in self.sets))
        elif method == "product":
            return " * ".join("(%s)" % s._to_c(name) for s in self.sets)
        elif method == "lukasiewicz":
            return "max(2, 0, %s - %d)" % (" + ".join(s._to_c(name) for s in self.sets), len(self.sets) - 1)
        else:
            raise ValueError("Invalid OR method in context: %s" % method)


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
        if x < self.a or x > self.c:
            return 0.0
        if x < self.b:
            if self.b == self.a:
                return 1.0
            return (x - self.a) / (self.b - self.a)
        if self.c == self.b:
            return 1.0
        return (self.c - x) / (self.c - self.b)

    def _to_c(self, name):
        return "triangular({a},{b},{c},{x})".format(x=name,
                                                    a=self.a,
                                                    b=self.b,
                                                    c=self.c)


class TrapezoidalFuzzySet(FuzzySet):
    """A fuzzy set defined by a trapezoidal function"""

    def __init__(self, a, b, c, d):
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
        if x < self.a or x > self.d:
            return 0.0
        if x < self.b:
            if self.b == self.a:
                return 1.0
            return (x - self.a) / (self.b - self.a)
        if x > self.c:
            if self.d == self.c:
                return 1.0
            return (self.d - x) / (self.d - self.c)
        return 1.0

    def _to_c(self, name):
        return "trapezoidal({a},{b},{c},{d},{x})".format(x=name,
                                                         a=self.a,
                                                         b=self.b,
                                                         c=self.c,
                                                         d=self.d)


def _gauss(x, s, a):
    return exp(-((x - a) / s) ** 2 / 2)


class GaussianFuzzySet(FuzzySet):
    """A fuzzy set defined by a Gaussian function

    .. math::

        G_{s,a}(x) = \\mathrm{e}^{-\\frac{{(x - a)}^2}{2 s^2}}

    Parameters:
        s: Width of the Gaussian.
        a: Position of the peak of the Gaussian.

    """

    def __init__(self, s, a):
        self.s = s
        self.a = a

        super().__init__()

    def _get_description(self):
        return {"type": "gaussian", "s": self.s, "a": self.a}

    @staticmethod
    def _from_description(description):
        return GaussianFuzzySet(description["s"], description["a"])

    def __call__(self, x):
        return _gauss(x, self.s, self.a)

    def _to_c(self, name):
        return "gauss({x}, {s}, {a})".format(x=name, s=self.s, a=self.a)


def _gauss2(x, s1, a1, s2, a2):
    if a1 <= x <= a2:
        return 1
    if x < a1:
        return _gauss(x, s1, a1)
    return _gauss(x, s2, a2)


class Gaussian2FuzzySet(FuzzySet):
    """
    A fuzzy set defined by two Gaussian functions

    .. math::


        G^2_{s_1,a_1,s_2,a_2}(x) =
                 \\begin{cases}
                   G_{s_1, a_1}(x), &\\quad\\text{if } x\\leq a_1\\\\
                   1, &\\quad\\text{if } a_1 \\leq x \\leq a_2\\\\
                   G_{s_2, a_2}(x), &\\quad\\text{if } x\\geq a_2\\\\
                 \\end{cases}

    Parameters:
        s1: Width of the first Gaussian.
        a1: Start of the membership=1.0 plateau.
        s2: Width of the second Gaussian.
        a2: End of the membership=1.0 plateau.

    """

    def __init__(self, s1, a1, s2, a2):
        assert a1 <= a2, "Positions must be ordered (a1 <= a2)"
        self.s1 = s1
        self.a1 = a1
        self.s2 = s2
        self.a2 = a2

        super().__init__()

    def _get_description(self):
        return {"type": "gaussian2", "s1": self.s1, "a1": self.a1, "s2": self.s2, "a2": self.a2}

    @staticmethod
    def _from_description(description):
        return Gaussian2FuzzySet(description["s1"], description["a1"], description["s2"], description["a2"])

    def __call__(self, x):
        return _gauss2(x, self.s1, self.a1, self.s2, self.a2)

    def _to_c(self, name):
        return "gauss2({x}, {s1}, {a1}, {s2}, {a2})".format(x=name, s1=self.s1, a1=self.a1, s2=self.s2, a2=self.a2)


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
              "bell": BellFuzzySet, "gaussian": GaussianFuzzySet, "gaussian2": Gaussian2FuzzySet,
              "triangular": TriangularFuzzySet, "trapezoidal": TrapezoidalFuzzySet}
