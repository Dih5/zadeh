import zadeh

import os


def test_parse():
    """Load an external .fis file"""
    fis = zadeh.FIS.from_matlab(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tipper.fis"))

    assert isinstance(fis.get_crisp_output({"food": 5, "service": 5}), float)
