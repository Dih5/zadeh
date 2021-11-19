from zadeh.sets import *

from math import isclose


def test_numbers():
    n = FuzzySet(lambda x: 0.5 if x == 1 else 0.1)
    assert n(1) == 0.5
    assert n(2) == 0.1

    n = SingletonSet(1)
    assert n(1) == 1.0
    assert n(2) == 0.0

    n = DiscreteFuzzySet({1: 1.0, 2: 0.1})
    assert n(1) == 1.0
    assert n(2) == 0.1
    assert n(3) == 0.0

    n = DiscreteFuzzySet({"spain": 1.0, "france": 0.1})
    assert n("spain") == 1.0
    assert n("france") == 0.1
    assert n("germany") == 0.0

    n = BellFuzzySet(10, 1, 50)
    assert n(50) == 1.0
    assert n(40) == 0.5


def test_limit_cases():
    """Test limit cases on piecewise linear functions"""
    n = TriangularFuzzySet(0, 0, 1)
    assert n(-1) == 0
    assert n(0) == 1
    assert abs(n(1)) < 1E-6

    n = TriangularFuzzySet(1, 1, 2)
    assert n(-1) == 0
    assert n(1) == 1
    assert abs(n(2)) < 1E-6

    n = TrapezoidalFuzzySet(0, 0, 1, 1)
    assert n(-1) == 0
    assert n(0) == 1
    assert isclose(n(0.5), 1)
    assert n(1) == 1
    assert n(2) == 0


def test_operation():
    """Test fuzzy set operations"""
    n = FuzzySet(lambda x: 0.5 if x == 1 else 0.1)
    n2 = -n
    assert isclose(n2(1), 0.5)
    assert isclose(n2(2), 0.9)

    n = FuzzySet(lambda x: 0.2 if x == 1 else 0.8)
    n3 = -n
    assert isclose(n2(1), 0.5)
    assert isclose(n2(2), 0.9)
    assert isclose(n3(1), 0.8)
    assert isclose(n3(2), 0.2)

    # OR / union
    n = FuzzySet(lambda x: 0.5 if x == 1 else 0.1) | FuzzySet(lambda x: 0.7 if x == 2 else 0.05)
    assert isclose(n(1), 0.5)
    assert isclose(n(2), 0.7)
    assert isclose(n(3), 0.1)

    # AND / intersection
    n = FuzzySet(lambda x: 0.5 if x == 1 else 0.1) & FuzzySet(lambda x: 0.7 if x == 2 else 0.05)
    assert isclose(n(1), 0.05)
    assert isclose(n(2), 0.1)
    assert isclose(n(3), 0.05)
