from mpi4py import MPI

import basix
import dolfinx.mesh
import numpy as np
import pytest
import ufl
from dolfinx import default_scalar_type, fem

import festim as F

test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
x = ufl.SpatialCoordinate(test_mesh.mesh)

element_CG = basix.ufl.element(
    basix.ElementFamily.P,
    test_mesh.mesh.basix_cell(),
    1,
    basix.LagrangeVariant.equispaced,
)
test_function_space = fem.functionspace(test_mesh.mesh, element_CG)
test_function = fem.Function(test_function_space)


@pytest.mark.parametrize(
    "value",
    [
        1,
        fem.Constant(test_mesh.mesh, default_scalar_type(1.0)),
        1.0,
        "coucou",
        2 * x[0],
    ],
)
def test_temperature_type_and_processing(value):
    """Test that the temperature type is correctly set"""

    if not isinstance(value, fem.Constant | int | float):
        with pytest.raises(TypeError):
            F.as_fenics_constant(value, test_mesh.mesh)
    else:
        assert isinstance(F.as_fenics_constant(value, test_mesh.mesh), fem.Constant)


@pytest.mark.parametrize(
    "input_value, expected_output_type", [(1.0, fem.Constant), (3, fem.Constant)]
)
def test_value_convert_float_int_inputs(input_value, expected_output_type):
    """Test that float and  value is correctly converted"""

    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)

    test_value = F.Value(input_value)

    test_value.convert_input_value(mesh=mesh)

    assert isinstance(test_value.fenics_object, expected_output_type)


@pytest.mark.parametrize(
    "input_value, expected_output_type",
    [
        (lambda t: t, fem.Constant),
        (lambda t: 1.0 + t, fem.Constant),
        (lambda x: 1.0 + x[0], ufl.core.expr.Expr),
        (lambda x, t: 1.0 + x[0] + t, ufl.core.expr.Expr),
        (lambda x, t, T: 1.0 + x[0] + t + T, ufl.core.expr.Expr),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
            ufl.core.expr.Expr,
        ),
    ],
)
def test_value_convert_up_to_ufl_inputs(input_value, expected_output_type):
    """Test that float and  value is correctly converted"""

    my_mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V = fem.functionspace(my_mesh, ("Lagrange", 1))
    my_t = fem.Constant(my_mesh, default_scalar_type(10))
    my_T = fem.Constant(my_mesh, default_scalar_type(3))

    test_value = F.Value(input_value)

    test_value.convert_input_value(
        mesh=my_mesh,
        function_space=V,
        t=my_t,
        temperature=my_T,
        up_to_ufl_expr=True,
    )

    assert isinstance(test_value.fenics_object, expected_output_type)


@pytest.mark.parametrize(
    "input_value, expected_output_type",
    [
        (lambda t: t, fem.Constant),
        (lambda t: 1.0 + t, fem.Constant),
        (lambda x: 1.0 + x[0], fem.Function),
        (lambda x, t: 1.0 + x[0] + t, fem.Function),
        (lambda x, t, T: 1.0 + x[0] + t + T, fem.Function),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
            ufl.core.expr.Expr,
        ),
    ],
)
def test_value_convert_callable_inputs(input_value, expected_output_type):
    """Test that float and  value is correctly converted"""

    my_mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 12)
    my_t = fem.Constant(my_mesh, default_scalar_type(8))
    my_T = fem.Constant(my_mesh, default_scalar_type(5))

    element_CG = basix.ufl.element(
        basix.ElementFamily.P,
        my_mesh.basix_cell(),
        1,
        basix.LagrangeVariant.equispaced,
    )
    my_function_space = fem.functionspace(my_mesh, element_CG)

    test_value = F.Value(input_value)

    test_value.convert_input_value(
        function_space=my_function_space,
        mesh=my_mesh,
        t=my_t,
        temperature=my_T,
    )

    assert isinstance(test_value.fenics_object, expected_output_type)


def test_error_raised_wehn_input_value_is_not_accepted():
    """Test that an error is raised when the input value is not accepted"""

    with pytest.raises(
        TypeError,
        match=(
            "Value must be a float, int, fem.Constant, np.ndarray, fem.Expression, "
            "ufl.core.expr.Expr, fem.Function, or callable not coucou"
        ),
    ):
        F.Value("coucou")


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        (1.0, False),
        (None, False),
        (fem.Constant(test_mesh.mesh, default_scalar_type(1.0)), False),
        (lambda t: t, True),
        (lambda t: 1.0 + t, True),
        (lambda x: 1.0 + x[0], False),
        (lambda x, t: 1.0 + x[0] + t, True),
        (lambda x, t, T: 1.0 + x[0] + t + T, True),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
            True,
        ),
    ],
)
def test_time_dependent_values(input_value, expected_output):
    """Test that the time_dependent attribute is correctly set"""

    test_value = F.Value(input_value)

    assert test_value.time_dependent == expected_output


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        (1.0, False),
        (None, False),
        (fem.Constant(test_mesh.mesh, default_scalar_type(1.0)), False),
        (lambda t: t, False),
        (lambda T: 1.0 + T, True),
        (lambda x: 1.0 + x[0], False),
        (lambda x, t: 1.0 + x[0] + t, False),
        (lambda x, t, T: 1.0 + x[0] + t + T, True),
        (
            lambda T, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + T[0], 0.0),
            True,
        ),
    ],
)
def test_temperature_dependent_values(input_value, expected_output):
    """Test that the time_dependent attribute is correctly set"""

    test_value = F.Value(input_value)

    assert test_value.temperature_dependent == expected_output


@pytest.mark.parametrize(
    "value",
    [
        fem.Constant(test_mesh.mesh, default_scalar_type(1.0)),
        test_function,
    ],
)
def test_input_values_of_constants_and_functions_are_accepted(value):
    """Test that the input values of constants and functions are accepted"""

    test_value = F.Value(value)

    test_value.convert_input_value()

    assert test_value.fenics_object == value


def test_input_values_of_expressions_are_accepted():
    """Test that the input values of constants and functions are accepted"""

    my_func = lambda x: 1.0 + x[0]
    kwargs = {}
    kwargs["x"] = x
    mapped_func = my_func(**kwargs)

    test_expression = fem.Expression(
        mapped_func,
        test_function_space.element.interpolation_points(),
    )
    test_value = F.Value(input_value=test_expression)

    test_value.convert_input_value()

    assert test_value.fenics_interpolation_expression == test_expression


def test_ValueError_raised_when_callable_returns_wrong_type():
    """The create_value_fenics method should raise a ValueError when the callable
    returns an object which is not a float or int"""

    def my_value(t):
        return ufl.conditional(ufl.lt(t, 0.5), 100, 0)

    test_value = F.Value(my_value)

    T = fem.Constant(test_mesh.mesh, 550.0)
    t = fem.Constant(test_mesh.mesh, 0.0)

    with pytest.raises(
        ValueError,
        match="self.value should return a float or an int, not <class 'ufl.conditional.Conditional'",
    ):
        test_value.convert_input_value(mesh=test_mesh.mesh, temperature=T, t=t)


@pytest.mark.parametrize(
    "value",
    [
        1,
        1.0,
        np.array([1.0, 2.0, 3.0]),
        lambda t: t,
        lambda T: 1.0 + T,
        lambda x: 1.0 + x[0],
        lambda x, t: 1.0 + x[0] + t,
        lambda x, t, T: 1.0 + x[0] + t + T,
        lambda T, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + T[0], 0.0),
    ],
)
def test_value_representation(value):
    """Test that the representation of the value is correct"""

    test_value = F.Value(value)

    assert repr(test_value) == f"{value}"
