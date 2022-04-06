import FESTIM
import fenics as f
import sympy as sp
import numpy as np


def test_implantation_flux_attributes():
    """
    Checks all attributes of the ImplantationFlux class
    """
    flux = 1
    imp_depth = 5e-9
    width = 5e-9
    distribution = 1 / (width * (2 * np.pi)**0.5) * \
        sp.exp(-0.5 * ((FESTIM.x - imp_depth) / width)**2)
    expected_value = sp.printing.ccode(flux * distribution)

    my_source = FESTIM.ImplantationFlux(flux=flux, imp_depth=imp_depth,
                                        width=width, volume=1)

    assert my_source.flux == flux
    assert my_source.imp_depth == imp_depth
    assert my_source.width == width
    assert my_source.value._cppcode == expected_value


def test_implantation_flux_with_time_dependancy():
    """
    Checks that ImplantationFlux has the correct value attribute when using
    time dependdant arguments
    """
    flux = 1 * (FESTIM.t < 10)
    imp_depth = 5e-9
    width = 5e-9
    distribution = 1 / (width * (2 * np.pi)**0.5) * \
        sp.exp(-0.5 * ((FESTIM.x - imp_depth) / width)**2)
    expected_value = sp.printing.ccode(flux * distribution)

    my_source = FESTIM.ImplantationFlux(flux=1 * (FESTIM.t < 10),
                                        imp_depth=5e-9, width=5e-9, volume=1)

    assert my_source.value._cppcode == expected_value


def test_source_with_float_value():
    """
    Tests that Source can be created with a float value and that the .value
    attribute is Constant
    """
    source = FESTIM.Source(2.0, volume=1, field="solute")
    assert isinstance(source.value, f.Constant)


def test_source_with_int_value():
    """
    Tests that Source can be created with a int value and that the .value
    attribute is Constant
    """
    source = FESTIM.Source(2, volume=1, field="solute")
    assert isinstance(source.value, f.Constant)


def test_source_with_expression_value():
    """
    Tests that Source can be created with a fenics.Expression value and that
    the .value attribute is fenics.Expression
    """
    value = f.Expression("2", degree=1)
    source = FESTIM.Source(value, volume=1, field="solute")
    assert isinstance(source.value, f.Expression)


def test_source_with_userexpression_value():
    """
    Tests that Source can be created with a fenics.UserExpression value and
    that the .value attribute is fenics.UserExpression
    """
    class CustomExpr(f.UserExpression):
        def __init__(self):
            super().__init__()

        def eval(self, value, x):
            value[0] = 1

    source = FESTIM.Source(CustomExpr(), volume=1, field="solute")
    assert isinstance(source.value, f.UserExpression)
