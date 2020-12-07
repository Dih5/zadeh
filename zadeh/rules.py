from .sets import FuzzySet


class FuzzyProposition:
    """A fuzzy-logic proposition"""

    def __init__(self):
        pass

    def __call__(self, values):
        """Evaluate the statement, returning a number in [0, 1]"""
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

    # TODO: More generalized fuzzy set operations could be defined

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
        return min(p(values) for p in self.proposition_list)

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
        return max(p(values) for p in self.proposition_list)

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

    def __call__(self, values):
        """Evaluate the rule, returning a fuzzy number"""
        # TODO: Add more methods
        # Mamdani inference
        cutoff = self.antecedent(values)
        return FuzzySet(lambda x: min(cutoff, self.consequent.variable[self.consequent.value](x))) * self.weight

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
    def _from_description(description, variables_dict):
        return FuzzyRuleSet([FuzzyRule._from_description(d, variables_dict) for d in description["rule_list"]])

    def __call__(self, values):
        """Evaluate the set of rules, returning a fuzzy number"""
        # TODO: Consider multiple output
        # TODO: Add more methods
        # Mamdani inference
        return FuzzySet.n_ary_or([rule(values) for rule in self.rule_list])

    def __getitem__(self, item):
        return self.rule_list[item]

    def __repr__(self):
        return "FuzzyRuleSet<%s>" % str(self)

    def __str__(self):
        return "\n".join(str(s) for s in self.rule_list)

    def __iter__(self):
        return iter(self.rule_list)
