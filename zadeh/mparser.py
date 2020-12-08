"""Matlab .fis-like parser"""
import re
import configparser

from . import sets, variables, domains, fis, rules


# For a general reference on the format cf:
# http://functionbay.com/documentation/onlinehelp/default.htm#!Documents/introductiontothefisfileformat.htm

def read_mfis(path, steps=100):
    """Parse a MATLAB® Fuzzy Inference System-like file"""
    config = configparser.ConfigParser(inline_comment_prefixes="%")
    config.read(path)
    num_inputs = int(config["System"]["numinputs"])
    num_outputs = int(config["System"]["numoutputs"])

    # Raise errors for non-supported options
    if num_outputs != 1:
        raise NotImplementedError("Only one output is supported")
    if config["System"]["AndMethod"] != "'min'":
        raise NotImplementedError("And method not implemented")
    if config["System"]["OrMethod"] != "'max'":
        raise NotImplementedError("Or method not implemented")
    if config["System"]["Type"] != "'mamdani'":
        raise NotImplementedError("Type of inference not implemented")
    if config["System"]["ImpMethod"] != "'min'":
        raise NotImplementedError("Implementation method not implemented")
    if config["System"]["AggMethod"] != "'max'":
        raise NotImplementedError("Aggregation method not implemented")
    if config["System"]["DefuzzMethod"] != "'centroid'":
        raise NotImplementedError("Defuzzification method not implemented")

    inputs = [parse_variable(config["Input%d" % i], steps) for i in range(1, num_inputs + 1)]
    output = parse_variable(config["Output1"], steps)

    rules_ = [parse_rule(*x, inputs, output) for x in config["Rules"].items()]

    return fis.FIS(inputs, rules.FuzzyRuleSet(rules_), output)


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

    if value_f == "trimf":
        return value_name, sets.TriangularFuzzySet(*pars)
    elif value_f == "trapmf":
        return value_name, sets.TrapezoidalFuzzySet(*pars)
    elif value_f == "gaussmf":
        return value_name, sets.GaussianFuzzySet(*pars)
    elif value_f == "sigmf":
        return value_name, sets.SigmoidalFuzzySet(*pars)
    elif value_f == "psigmf":
        return value_name, sets.SigmoidalProductFuzzySet(*pars)
    elif value_f == "dsigmf":
        return value_name, sets.SigmoidalDifferenceFuzzySet(*pars)
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
