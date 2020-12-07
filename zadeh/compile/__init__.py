"""Compilation tools for Zadeh"""
import tempfile
import subprocess
import os
import ctypes

import jinja2

from ..fis import FIS
from ..sets import FuzzySet

jinja = jinja2.Environment(loader=jinja2.PackageLoader('zadeh', 'compile/templates'))

# TODO: Expose this configuration
CC = "gcc"
FLAGS = '-Wall -O3 -std=c99'
LDFLAGS = '-shared -lm'

__compiled_dir = "/tmp"


def compile_model(model, function_name="f"):
    """
    Generate and link a C-function.

    Args:
        model (FIS): Fuzzy inference System.
        function_name (str): Internal name of the function. Irrelevant while using the returned wrapper.
        CC (str): Compiler to use.

    Returns:
        Callable: a wrapper around the compiled function.
    """
    code = model._to_c()

    template = jinja.get_template("model.c")
    code = template.render(code=code, name=function_name, target=model.target.name,
                           inputs_listed=", ".join(x.name for x in model.variables),
                           inputs_typed=", ".join("double %s" % x.name for x in model.variables))

    with tempfile.NamedTemporaryFile("w", suffix=".c") as f:
        # TODO: Improve lib path selection. Not sure if uniqueness is guaranteed in this way.
        lib_path = os.path.join(__compiled_dir, os.path.basename(f.name)[:-2] + ".so")
        f.write(code)
        f.seek(0)
        proc = subprocess.run("%s %s %s -o %s %s" % (CC, FLAGS, LDFLAGS, lib_path, f.name), shell=True)

    proc.check_returncode()

    dll = ctypes.CDLL(lib_path)
    f = getattr(dll, function_name)
    f.argtypes = tuple([ctypes.c_double] + [ctypes.c_double for _ in model.variables])
    f.restype = ctypes.c_double

    f_crisp = getattr(dll, function_name + "_crisp")
    f_crisp.argtypes = tuple(
        [ctypes.c_double, ctypes.c_double, ctypes.c_int] + [ctypes.c_double for _ in model.variables])
    f_crisp.restype = ctypes.c_double

    return f, f_crisp


class CompiledFIS(FIS):
    """A compiled version of a FIS"""

    def __init__(self, variables, rules, target):
        super().__init__(variables, rules, target)

        self.f, self.f_crisp = compile_model(self)

    @staticmethod
    def from_existing(fis):
        """
        Get a CompiledFIS from a existing FIS.

        Args:
            fis (FIS): A Fuzzy Inference System

        Returns:
            CompiledFIS: A compiled version of the FIS
        """
        return CompiledFIS(fis.variables, fis.rules, fis.target)

    def get_output(self, values):
        return FuzzySet(lambda x: self.f(x, *self.dict_to_ordered(values)))

    def get_crisp_output(self, values):
        return self.f_crisp(self.target.domain.min, self.target.domain.max, self.target.domain.steps,
                            *self.dict_to_ordered(values))
