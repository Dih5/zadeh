from . import get_active_context
from .sets import FuzzySet, FuzzySetOr

try:
    from math import prod  # Python >= 3.8
except ImportError:
    def prod(xx):
        result = 1
        for x in xx:
            result *= x
        return result


class FuzzyProposition:
    """A fuzzy-logic proposition"""

    def __init__(self):
        pass

    def __call__(self, values):
        """Evaluate the statement, returning a number in [0, 1]"""
        raise NotImplementedError

    def _to_c(self):
        raise NotImplementedError

    def _get_description(self):
        raise NotImplementedError

    @staticmethod
    def _from_description(description, variables_dict):
        return _fuzzy_propositions[description["type"]]._from_description(description, variables_dict)

    def __repr__(self):
        return "FuzzyProposition<%s>" % str(self)

    def __str__(self):
        return "[Undefined proposition]"

    def __or__(self, other):
        return FuzzyOr([self, other])

    def __and__(self, other):
        return FuzzyAnd([self, other])

    def __invert__(self):
        return FuzzyNot(self)

    def __rshift__(self, other):
        if not isinstance(other, FuzzyProposition):
            raise ValueError("Implications can only be constructed with a proposition as consequent")
        return FuzzyRule(self, other)


class FuzzyValuation(FuzzyProposition):
    """An elemental fuzzy proposition of the form '<variable> is <value>'"""

    def __init__(self, variable, value):
        super().__init__()
        self.variable = variable
        self.value = value

    def _get_description(self):
        return {"type": "is", "variable": self.variable.name, "value": self.value}

    @staticmethod
    def _from_description(description, variables_dict):
        return FuzzyValuation(variables_dict[description["variable"]], description["value"])

    def __call__(self, values):
        return self.variable[self.value](values[self.variable.name])

    def _to_c(self):
        return self.variable[self.value]._to_c(self.variable.name)

    def __str__(self):
        return "%s is %s" % (self.variable.name, self.value)


class FuzzyNotValuation(FuzzyProposition):
    """An 'elemental' fuzzy proposition of the form <variable> is not <value>

    While this could be described using the unitary negation operator, the class is provided for convenience

    """

    def __init__(self, variable, value):
        super().__init__()
        self.variable = variable
        self.value = value

    def _get_description(self):
        return {"type": "is not", "variable": self.variable.name, "value": self.value}

    @staticmethod
    def _from_description(description, variables_dict):
        return FuzzyNotValuation(variables_dict[description["variable"]], description["value"])

    def __call__(self, values):
        return 1 - self.variable[self.value](values[self.variable.name])

    def _to_c(self):
        return "1 - (%s)" % self.variable[self.value]._to_c(self.variable.name)

    def __str__(self):
        return "%s is not %s" % (self.variable.name, self.value)


class FuzzyNot(FuzzyProposition):
    """A fuzzy proposition of the form 'not <p>'"""

    def __init__(self, proposition):
        super().__init__()
        self.proposition = proposition

    def _get_description(self):
        return {"type": "not", "children": [self.proposition._get_description()]}

    @staticmethod
    def _from_description(description, variables_dict):
        return FuzzyNot(variables_dict[description["children"][0]])

    def __call__(self, values):
        return 1 - self.proposition(values)

    def _to_c(self):
        return "1 - (%s)" % self.proposition._to_c()

    def __str__(self):
        return "not (%s)" % str(self.proposition)


class FuzzyAnd(FuzzyProposition):
    """A fuzzy proposition of the form <p1> and <p2>"""

    def __init__(self, proposition_list):
        super().__init__()
        self.proposition_list = proposition_list

    def _get_description(self):
        return {"type": "and", "children": [p._get_description() for p in self.proposition_list]}

    @staticmethod
    def _from_description(description, variables_dict):
        return FuzzyAnd([variables_dict[variable] for variable in description["children"]])

    def __call__(self, values):
        method = get_active_context().AND
        if method == "min":
            return min(p(values) for p in self.proposition_list)
        elif method == "product":
            return prod(p(values) for p in self.proposition_list)
        elif method == "lukasiewicz":
            return max(0, sum(p(values) for p in self.proposition_list) - (len(self.proposition_list) - 1))
        else:
            raise ValueError("Invalid AND method in context: %s" % method)

    def _to_c(self):
        method = get_active_context().AND
        if method == "min":
            return "min(%d, %s)" % (len(self.proposition_list), ", ".join(s._to_c() for s in self.proposition_list))
        elif method == "product":
            return " * ".join("%s" % s._to_c() for s in self.proposition_list)
        elif method == "lukasiewicz":
            return "max(2, 0, %s - %d)" % (
                " + ".join(s._to_c() for s in self.proposition_list), len(self.proposition_list) - 1)
        else:
            raise ValueError("Invalid OR method in context: %s" % method)

    def __str__(self):
        return " and ".join("(%s)" % str(p) for p in self.proposition_list)


class FuzzyOr(FuzzyProposition):
    """A fuzzy proposition of the form <p1> or <p2>"""

    def __init__(self, proposition_list):
        super().__init__()
        self.proposition_list = proposition_list

    def _get_description(self):
        return {"type": "or", "children": [p._get_description() for p in self.proposition_list]}

    @staticmethod
    def _from_description(description, variables_dict):
        return FuzzyOr(
            [FuzzyProposition._from_description(variable, variables_dict) for variable in description["children"]])

    def __call__(self, values):
        method = get_active_context().OR
        if method == "max":
            return max(p(values) for p in self.proposition_list)
        elif method == "psum":
            return 1 - prod(1 - p(values) for p in self.proposition_list)
        elif method == "bsum":
            return min(1, sum(p(values) for p in self.proposition_list))
        else:
            raise ValueError("Invalid OR method in context: %s" % method)

    def _to_c(self):
        method = get_active_context().OR
        if method == "max":
            return "max(%d, %s)" % (len(self.proposition_list), ", ".join([p._to_c() for p in self.proposition_list]))
        elif method == "psum":
            return "1 - %s" % " * ".join("(1 - %s)" % p._to_c() for p in self.proposition_list)
        elif method == "bsum":
            return "min(2, 1, %s)" % " + ".join(p._to_c() for p in self.proposition_list)
        else:
            raise ValueError("Invalid OR method in context: %s" % method)

    def __str__(self):
        return " or ".join("(%s)" % str(p) for p in self.proposition_list)


_fuzzy_propositions = {"or": FuzzyOr, "and": FuzzyAnd, "not": FuzzyNot, "is": FuzzyValuation,
                       "is not": FuzzyNotValuation}


class FuzzyRule:
    """A fuzzy rule of the form 'if <antecedent> then <consequent>', possibly with a weight in (0, 1]"""

    def __init__(self, antecedent, consequent, weight=1.0):
        assert 0 < weight <= 1.0, "weight must be in (0, 1]"
        super().__init__()
        self.antecedent = antecedent
        self.consequent = consequent
        self.weight = weight

        if not isinstance(consequent, FuzzyValuation):
            # TODO: Support this
            raise ValueError("Complex consequent rules not supported")

    def _get_description(self):
        return {"antecedent": self.antecedent._get_description(), "consequent": self.consequent._get_description(),
                "weight": self.weight}

    @staticmethod
    def _from_description(description, variables_dict):
        return FuzzyRule(FuzzyProposition._from_description(description["antecedent"], variables_dict),
                         FuzzyProposition._from_description(description["consequent"], variables_dict),
                         weight=description["weight"])

    def _to_c(self):
        method = get_active_context().implication
        if method == "min":
            output_code = "min(2, %s, %s)" % (self.antecedent._to_c(), self.consequent._to_c())
        elif method == "prod":
            output_code = "((%s) * (%s))" % (self.antecedent._to_c(), self.consequent._to_c())
        else:
            raise ValueError("Invalid implication method in context: %s" % method)

        if self.weight == 1.0:  # Simplify output
            return output_code
        return "%s * %f" % (output_code, self.weight)

    def __call__(self, values):
        """Evaluate the rule, returning a fuzzy number"""

        # Mamdani inference
        antecendent = self.antecedent(values)

        method = get_active_context().implication
        if method == "min":
            output_set = FuzzySet(lambda x: min(antecendent, self.consequent.variable[self.consequent.value](x)))

        elif method == "prod":
            output_set = FuzzySet(lambda x: antecendent * self.consequent.variable[self.consequent.value](x))

        else:
            raise ValueError("Invalid implication method in context: %s" % method)

        return output_set * self.weight

    def __repr__(self):
        return "FuzzyRule<%s>" % str(self)

    def __str__(self):
        return "if (%s) then (%s) [%f]" % (self.antecedent, self.consequent, self.weight)


class FuzzyRuleSet:
    """A set of fuzzy rules"""

    def __init__(self, rule_list):
        super().__init__()
        self.rule_list = rule_list

    def _get_description(self):
        return {"rule_list": [r._get_description() for r in self.rule_list]}

    @staticmethod
    def automatic(antecedent_var, consequent_var, weight=1.0, reverse=False):
        return FuzzyRuleSet(_autorules(antecedent_var, consequent_var, weight=weight, reverse=reverse))

    @staticmethod
    def _from_description(description, variables_dict):
        return FuzzyRuleSet([FuzzyRule._from_description(d, variables_dict) for d in description["rule_list"]])

    def _to_c(self):
        method = get_active_context().aggregation
        if method == "max":
            return "max(%d, %s)" % (len(self.rule_list), ", ".join(rule._to_c() for rule in self.rule_list))
        elif method == "psum":
            return "1 - %s" % " * ".join("(1 - %s)" % rule._to_c() for rule in self.rule_list)
        elif method == "bsum":
            return "min(2, 1, %s)" % " + ".join("(%s)" % rule._to_c() for rule in self.rule_list)
        else:
            raise ValueError("Invalid aggregation method in context: %s" % method)

    def __call__(self, values):
        """Evaluate the set of rules, returning a fuzzy number"""
        # Note the aggregation method might be different from the OR method
        method = get_active_context().aggregation
        if method not in ["max", "psum", "bsum"]:
            # Check now to distinguish errors in OR or aggregation
            raise ValueError("Invalid aggregation method in context: %s" % method)
        return FuzzySetOr([rule(values) for rule in self.rule_list], method=method)

    def __getitem__(self, item):
        return self.rule_list[item]

    def __repr__(self):
        return "FuzzyRuleSet<%s>" % str(self)

    def __str__(self):
        return "\n".join(str(s) for s in self.rule_list)

    def __iter__(self):
        return iter(self.rule_list)


# Automatic rules
def _ordered_values(v):
    """Get the values sorted by increasing centroid"""
    return [a[0] for a in
            sorted([(name, v.domain.centroid(fuzzy_set)) for name, fuzzy_set in v.values.items()], key=lambda x: x[1])]


def _autorules(antecedent_var, consequent_var, weight=1.0, reverse=False):
    v1 = _ordered_values(antecedent_var)
    v2 = _ordered_values(consequent_var)
    n = len(v1)
    m = len(v2)
    if n == m:
        pass
    elif n == m + 1 and n % 2:
        # Drop middle of v1
        v1.pop(n // 2)
        n -= 1
    elif n + 1 == m and m % 2:
        # Drop middle of v2
        v2.pop(m // 2)
        m -= 1
    else:
        raise ValueError("Unable to automatically choose a %d to %d mapping" % (n, m))
    if reverse:
        v2 = v2[::-1]

    return [FuzzyRule(antecedent_var == a, consequent_var == c, weight=weight) for a, c in zip(v1, v2)]
