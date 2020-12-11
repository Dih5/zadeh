"""Matlab .fis-like parser"""
import re
import configparser

from . import sets, variables, domains, fis, rules

# For a general reference on the format cf:
# http://functionbay.com/documentation/onlinehelp/default.htm#!Documents/introductiontothefisfileformat.htm

# Aggregation method converter
_aggregation_methods = {"max": "max", "sum": "bsum", "probor": "psum"}

# Implication method converter
_implication_methods = {"min": "min", "prod": "prod"}

# OR method converter
_OR_methods = {"max": "max", "sum": "bsum", "probor": "psum"}
# Sum does not appear in the docs, but we don't mind adding

# AND method converter
_AND_methods = {"min": "min", "prod": "prod"}


def read_mfis(path, steps=100):
    """Parse a MATLAB® Fuzzy Inference System-like file"""
    config = configparser.ConfigParser(inline_comment_prefixes="%")
    config.read(path)
    num_inputs = int(config["System"]["numinputs"])
    num_outputs = int(config["System"]["numoutputs"])

    # Raise errors for non-supported options
    if num_outputs != 1:
        raise NotImplementedError("Only one output is supported")
    if config["System"]["Type"] != "'mamdani'":
        raise NotImplementedError("Type of inference not implemented")

    inputs = [parse_variable(config["Input%d" % i], steps) for i in range(1, num_inputs + 1)]
    output = parse_variable(config["Output1"], steps)

    rules_ = [parse_rule(*x, inputs, output) for x in config["Rules"].items()]

    defuzzification = config["System"]["DefuzzMethod"][1:-1]
    if defuzzification not in ["centroid", "bisector", "som", "mom", "lom"]:
        raise ValueError("Invalid defuzzification: %s" % defuzzification)

    aggregation = config["System"]["AggMethod"][1:-1]
    try:
        aggregation = _aggregation_methods[aggregation]
    except KeyError as e:
        raise ValueError("Invalid aggregation: %s" % aggregation) from e

    implication = config["System"]["ImpMethod"][1:-1]
    try:
        implication = _implication_methods[implication]
    except KeyError as e:
        raise ValueError("Invalid implication: %s" % implication) from e

    AND = config["System"]["AndMethod"][1:-1]
    try:
        AND = _AND_methods[AND]
    except KeyError as e:
        raise ValueError("Invalid AND method: %s" % AND) from e

    OR = config["System"]["OrMethod"][1:-1]
    try:
        OR = _OR_methods[OR]
    except KeyError as e:
        raise ValueError("Invalid OR method: %s" % OR) from e

    return fis.FIS(inputs, rules.FuzzyRuleSet(rules_), output, defuzzification=defuzzification, aggregation=aggregation,
                   implication=implication, AND=AND, OR=OR)


def parse_variable(variable, steps=100):
    """
    Parse a section defining a fuzzy variable

    Args:
        variable: Section of the file defining the variable.
        steps (int): The number of steps for FloatDomain.

    Returns:
        FuzzyVariable: A representation of the variable

    """
    name = variable["Name"][1:-1]
    range_ = [float(x) for x in variable["Range"][1:-1].split()]
    mfs = [variable["MF%d" % j] for j in range(1, int(variable["NumMFs"]) + 1)]

    v = variables.FuzzyVariable(domains.FloatDomain(name, *range_, steps),
                                dict([parse_mf(mf) for mf in mfs])
                                )

    return v


# Mapping from Matlab MF to the equivalent FuzzySet class (ordered parameters must be equivalent as well)
_direct_equivalence_sets = {
    "trimf": sets.TriangularFuzzySet,
    "trapmf": sets.TrapezoidalFuzzySet,
    "gaussmf": sets.GaussianFuzzySet,
    "gauss2mf": sets.Gaussian2FuzzySet,
    "sigmf": sets.SigmoidalFuzzySet,
    "psigmf": sets.SigmoidalProductFuzzySet,
    "dsigmf": sets.SigmoidalDifferenceFuzzySet,
    "smf": sets.SFuzzySet,
    "zmf": sets.ZFuzzySet,
    "pimf": sets.PiFuzzySet,
}


def parse_mf(description):
    """
    Parse a membership function

    Args:
        description (str): A line defining the membership function.

    Returns:
        FuzzySet: A fuzzy set defined by the membership function

    """
    value_name, value_f, pars = re.match(r"'(.*)':'(.*)',\[(.*)\]", description).groups()
    pars = [float(x) for x in pars.split()]

    if value_f in _direct_equivalence_sets:
        return value_name, _direct_equivalence_sets[value_f](*pars)
    else:
        raise ValueError("Unknown membership function: %s" % value_f)


def parse_rule(rule, operation, inputs, output):
    """
    Parse a line defining a fuzzy rule

    Args:
        rule (str): Line defining the rule
        operation (str): "1" for "and", "2" for "or" (file format meaning).
        inputs (list of FuzzyVariable): The ordered list of inputs.
        output (FuzzyVariable): The output of the system

    Returns:
        FuzzyRule: The description of the fuzzy rule

    """
    input_values, target_value, weight = re.match(r"(.*), (.*) \((.*)\)", rule).groups()

    values = [int(x) for x in input_values.split()]
    weight = float(weight)
    target_value = int(target_value)

    # FIXME: The value position depends on the dict implementation preserving order
    lhs = [(rules.FuzzyValuation if v > 0 else rules.FuzzyNotValuation)(var, list(var.values)[abs(v) - 1]) for var, v in
           zip(inputs, values)]
    lhs = {"1": rules.FuzzyAnd, "2": rules.FuzzyOr}[operation](lhs)
    rhs = (rules.FuzzyValuation if target_value > 0 else rules.FuzzyNotValuation)(output, list(output.values)[
        abs(target_value) - 1])
    return rules.FuzzyRule(lhs, rhs, weight=weight)
