# Copyright 2026 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0.

"""Verify .pyi stubs match the runtime API to prevent drift."""

import hola_opt as hola

EXPECTED_CLASSES = [
    "Real",
    "Integer",
    "Categorical",
    "Minimize",
    "Maximize",
    "Gmm",
    "Sobol",
    "Random",
    "Space",
    "Study",
    "Trial",
    "CompletedTrial",
    "dashboard_dir",
]

EXPECTED_STUDY_METHODS = [
    "ask",
    "tell",
    "cancel",
    "top_k",
    "pareto_front",
    "trials",
    "trial_count",
    "update_objectives",
    "save",
    "run",
    "serve",
    "connect",
    "load",
]


def test_all_classes_exported():
    """Every class in the stubs must exist at runtime."""
    for name in EXPECTED_CLASSES:
        assert hasattr(hola, name), f"Missing export: hola.{name}"


def test_study_methods_exist():
    """Every method in Study stubs must exist at runtime."""
    for method in EXPECTED_STUDY_METHODS:
        assert hasattr(hola.Study, method), f"Missing method: Study.{method}"


def test_all_in_module_all():
    """__all__ must list every expected class."""
    for name in EXPECTED_CLASSES:
        assert name in hola.__all__, f"{name} not in hola.__all__"


def test_property_access():
    """Spot-check that stubbed properties are accessible."""
    r = hola.Real(0.0, 1.0)
    assert r.min == 0.0
    assert r.max == 1.0
    assert r.scale == "linear"

    i = hola.Integer(1, 10)
    assert i.min == 1
    assert i.max == 10

    c = hola.Categorical(["a", "b"])
    assert c.choices == ["a", "b"]

    m = hola.Minimize("loss", priority=2.0)
    assert m.field == "loss"
    assert m.priority == 2.0
    assert m.target is None
