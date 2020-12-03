import zadeh


def test_description():
    """Test descriptions are working"""

    # Example FIS
    service = zadeh.FuzzyVariable(
        zadeh.FloatDomain("service", 0, 10, 100),
        {
            "poor": zadeh.GaussianFuzzySet(1.5, 0),
            "good": zadeh.GaussianFuzzySet(1.5, 5),
            "excellent": zadeh.GaussianFuzzySet(1.5, 10),
        },
    )

    food = zadeh.FuzzyVariable(
        zadeh.FloatDomain("food", 0, 10, 100),
        {
            "rancid": zadeh.TrapezoidalFuzzySet(-2, 0, 1, 3),
            "delicious": zadeh.TrapezoidalFuzzySet(7, 9, 10, 12),
        },
    )
    tip = zadeh.FuzzyVariable(
        zadeh.FloatDomain("tip", 0, 30, 100),
        {
            "cheap": zadeh.TriangularFuzzySet(0, 5, 10),
            "average": zadeh.TriangularFuzzySet(10, 15, 20),
            "generous": zadeh.TriangularFuzzySet(20, 25, 30),
        },
    )

    rule_set = zadeh.FuzzyRuleSet(
        [
            zadeh.FuzzyRule(
                zadeh.FuzzyValuation(service, "poor")
                | zadeh.FuzzyValuation(food, "rancid"),
                zadeh.FuzzyValuation(tip, "cheap"),
            ),
            zadeh.FuzzyRule(
                zadeh.FuzzyValuation(service, "good"), zadeh.FuzzyValuation(tip, "average"),
            ),
            zadeh.FuzzyRule(
                zadeh.FuzzyValuation(service, "excellent")
                | zadeh.FuzzyValuation(food, "delicious"),
                zadeh.FuzzyValuation(tip, "generous"),
            ),
        ]
    )

    fis = zadeh.FIS([food, service], rule_set, tip)

    # Copy using the description and check descriptions are alike
    d = fis._get_description()
    assert d == zadeh.FIS._from_description(d)._get_description()
