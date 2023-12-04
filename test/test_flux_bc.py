import numpy as np
import pytest
import ufl
from ufl.conditional import Conditional
from dolfinx import fem
import dolfinx.mesh
from mpi4py import MPI
import festim as F

dummy_mat = F.Material(D_0=1, E_D=1, name="dummy_mat")
mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)


def test_init():
    """Test that the attributes are set correctly"""
    # create a DirichletBC object
    subdomain = F.SurfaceSubdomain1D(1, x=0)
    value = 1.0
    species = "test"
    bc = F.FluxBC(subdomain, value, species)

    # check that the attributes are set correctly
    assert bc.subdomain == subdomain
    assert bc.value == value
    assert bc.species == species
    assert bc.value_fenics is None
    assert bc.bc_expr is None


@pytest.mark.parametrize(
    "value, expected_type",
    [
        (1.0, fem.Constant),
        (lambda t: t, fem.Constant),
        (lambda t: 1.0 + t, fem.Constant),
        (lambda x: 1.0 + x[0], fem.Function),
        (lambda x, t: 1.0 + x[0] + t, fem.Function),
        (lambda x, t, T: 1.0 + x[0] + t + T, fem.Function),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
            fem.Function,
        ),
        (lambda t: 100.0 if t < 1 else 0.0, fem.Constant),
    ],
)
def test_create_value_fenics_type(value, expected_type):
    """Test that"""
    # BUILD
    left = F.SurfaceSubdomain1D(1, x=0)
    my_species = F.Species("test")
    my_func_space = fem.FunctionSpace(mesh, ("P", 1))
    T = F.as_fenics_constant(1, mesh)
    t = F.as_fenics_constant(0, mesh)
    bc = F.FluxBC(subdomain=left, value=value, species=my_species)

    # RUN
    bc.create_value_fenics(mesh, my_func_space, T, t)

    # TEST
    # check that the value_fenics attribute is set correctly
    assert isinstance(bc.value_fenics, expected_type)


@pytest.mark.parametrize(
    "value, expected_value",
    [
        (1.0, 1.0),
        (lambda t: t, 0.0),
        (lambda t: 4.0 + t, 4.0),
        (lambda x: 1.0 + x[0], 2.0),
        (lambda x, t: 3.0 + x[0] + t, 4.0),
        (lambda x, t, T: 2.0 + x[0] + t + T, 4.0),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 50.0 + x[0], 0.0),
            51,
        ),
        (lambda t: 50.0 if t < 1 else 0.0, 50),
    ],
)
def test_create_value_fenics_value(value, expected_value):
    """Test that"""
    # BUILD
    left = F.SurfaceSubdomain1D(1, x=0)
    my_species = F.Species("test")
    my_func_space = fem.FunctionSpace(mesh, ("P", 1))
    T = F.as_fenics_constant(1, mesh)
    t = F.as_fenics_constant(0, mesh)
    bc = F.FluxBC(subdomain=left, value=value, species=my_species)

    # RUN
    bc.create_value_fenics(mesh, my_func_space, T, t)

    # TEST
    # check that the value_fenics attribute is set correctly
    if isinstance(bc.value_fenics, fem.Constant):
        assert np.isclose(bc.value_fenics.value, expected_value)

    if isinstance(bc.value_fenics, fem.Function):
        assert np.isclose(bc.value_fenics.x.array[-1], expected_value)


def test_value_fenics_setter_error():
    left = F.SurfaceSubdomain1D(1, x=0)
    my_species = F.Species("test")
    bc = F.FluxBC(subdomain=left, value=1.0, species=my_species)

    with pytest.raises(
        TypeError,
        match="Value must be a dolfinx.fem.Function, dolfinx.fem.Constant, or a np.ndarray not <class 'str'>",
    ):
        bc.value_fenics = "coucou"


def test_ValueError_raised_when_callable_returns_wrong_type():
    """The create_value_fenics method should raise a ValueError when the callable
    returns an object which is not a float or int"""

    surface = F.SurfaceSubdomain(id=1)
    species = F.Species("test")

    def my_value(t):
        return ufl.conditional(ufl.lt(t, 0.5), 100, 0)

    bc = F.FluxBC(subdomain=surface, value=my_value, species=species)

    my_function_space = fem.FunctionSpace(mesh, ("CG", 1))
    T = fem.Constant(mesh, 550.0)
    t = fem.Constant(mesh, 0.0)

    with pytest.raises(
        ValueError,
        match="self.value should return a float or an int, not <class 'ufl.conditional.Conditional'",
    ):
        bc.create_value_fenics(mesh, my_function_space, T, t)


@pytest.mark.parametrize(
    "input, expected_value",
    [
        (1.0, False),
        (None, False),
        (fem.Constant(mesh, 1.0), False),
        (lambda t: t, True),
        (lambda t: 1.0 + t, True),
        (lambda x: 1.0 + x[0], False),
        (lambda x, t: 1.0 + x[0] + t, True),
        (lambda x, t, T: 1.0 + x[0] + t + T, True),
        (lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0), True),
    ],
)
def test_bc_time_dependent_attribute(input, expected_value):
    """Test that the time_dependent attribute is correctly set"""
    surface = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    my_species = F.Species("test")
    my_bc = F.FluxBC(subdomain=surface, value=input, species=my_species)

    assert my_bc.time_dependent is expected_value


@pytest.mark.parametrize(
    "input, expected_value",
    [
        (1.0, False),
        (None, False),
        (fem.Constant(mesh, 1.0), False),
        (lambda T: T, True),
        (lambda t: 1.0 + t, False),
        (lambda x, T: 1.0 + x[0] + T, True),
        (lambda x, t, T: 1.0 + x[0] + t + T, True),
        (lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0), False),
    ],
)
def test_bc_temperature_dependent_attribute(input, expected_value):
    """Test that the temperature_dependent attribute is correctly set"""
    surface = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    my_species = F.Species("test")
    my_bc = F.FluxBC(subdomain=surface, value=input, species=my_species)

    assert my_bc.temperature_dependent is expected_value
