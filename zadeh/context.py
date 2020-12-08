"""Context managers for changing the meaning of the fuzzy operations"""
from contextlib import contextmanager


class FuzzyContext:
    """Define contextual information on how the fuzzy operations are formalized"""

    def __init__(self, defuzzification="centroid"):
        self.defuzzification = defuzzification


# Global variable set by the context manager. Note references to it are getting when import is made, so it cannot be used directly
_active_context = FuzzyContext()


def get_active_context():
    """Return the active FuzzyContext"""
    return _active_context


@contextmanager
def set_fuzzy_context(new_context):
    """Set a FuzzyContext as active"""
    global _active_context
    store = _active_context
    _active_context = new_context
    try:
        yield
    finally:
        _active_context = store
