from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import pytest
import ufl
from dolfinx import default_scalar_type, fem
from ufl.conditional import Conditional

import festim as F

dummy_mat = F.Material(D_0=1, E_D=1, name="dummy_mat")

mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)


def test_init():
    """Test that the attributes are set correctly"""
    # create a DirichletBC object
    subdomain = F.SurfaceSubdomain1D(1, x=0)
    value = 1.0
    species = "test"
    bc = F.DirichletBC(subdomain, value, species)

    # check that the attributes are set correctly
    assert bc.subdomain == subdomain
    assert bc.value == value
    assert bc.species == species
    assert bc.value_fenics is None
    assert bc.bc_expr is None


def test_value_fenics():
    """Test that the value_fenics attribute can be set to a valid value
    and that an invalid type throws an error
    """
    # create a DirichletBC object
    subdomain = F.SurfaceSubdomain1D(1, x=0)
    value = 1.0
    species = "test"
    bc = F.DirichletBC(subdomain, value, species)

    # set the value_fenics attribute to a valid value
    value_fenics = fem.Constant(mesh, 2.0)
    bc.value_fenics = value_fenics

    # check that the value_fenics attribute is set correctly
    assert bc.value_fenics == value_fenics

    # set the value_fenics attribute to an invalid value
    with pytest.raises(TypeError):
        bc.value_fenics = "invalid"


def test_callable_for_value():
    """Test that the value attribute can be a callable function of x and t"""

    subdomain = F.SurfaceSubdomain1D(1, x=1)
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)

    def value(x, t):
        return 1.0 + x[0] + t

    species = F.Species("test")

    bc = F.DirichletBC(subdomain, value, species)

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[subdomain, vol_subdomain],
        species=[species],
    )

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()

    T = fem.Constant(my_model.mesh.mesh, 550.0)
    t = fem.Constant(my_model.mesh.mesh, 0.0)
    bc.create_value(my_model.function_space, T, t)

    # check that the value_fenics attribute is set correctly
    assert isinstance(bc.value_fenics, fem.Function)

    # check the initial value of the boundary condition
    assert bc.value_fenics.x.petsc_vec.array[-1] == float(
        value(x=np.array([subdomain.x]), t=0.0)
    )

    # check the value of the boundary condition after updating the time
    for i in range(10):
        t.value = i
        bc.update(float(t))
        expected_value = float(value(x=np.array([subdomain.x]), t=float(t)))
        computed_value = bc.value_fenics.x.petsc_vec.array[-1]
        assert np.isclose(computed_value, expected_value)


def test_value_callable_x_t_T():
    """Test that the value attribute can be a callable function of x, t, and T"""

    subdomain = F.SurfaceSubdomain1D(1, x=1)
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)

    def value(x, t, T):
        return 1.0 + x[0] + t + T

    species = F.Species("test")

    bc = F.DirichletBC(subdomain, value, species)

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[subdomain, vol_subdomain],
        species=[species],
    )

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()

    T = fem.Constant(my_model.mesh.mesh, 550.0)
    t = fem.Constant(my_model.mesh.mesh, 0.0)
    bc.create_value(my_model.function_space, T, t)

    # check that the value_fenics attribute is set correctly
    assert isinstance(bc.value_fenics, fem.Function)

    # check the initial value of the boundary condition
    assert np.isclose(
        bc.value_fenics.x.petsc_vec.array[-1],
        float(value(x=np.array([subdomain.x]), t=float(t), T=float(T))),
    )

    # check the value of the boundary condition after updating the time
    for i in range(10):
        t.value = i
        T.value += i
        bc.update(float(t))

        expected_value = float(value(x=np.array([subdomain.x]), t=float(t), T=float(T)))
        computed_value = bc.value_fenics.x.petsc_vec.array[-1]
        assert np.isclose(computed_value, expected_value)


@pytest.mark.parametrize("value", [lambda t: t, lambda t: 1.0 + t])
def test_callable_t_only(value):
    """Test that the value attribute can be a callable function of t only"""

    subdomain = F.SurfaceSubdomain1D(1, x=1)
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    species = F.Species("test")

    bc = F.DirichletBC(subdomain, value, species)

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[subdomain, vol_subdomain],
        species=[species],
    )

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()

    T = fem.Constant(my_model.mesh.mesh, 550.0)
    t = fem.Constant(my_model.mesh.mesh, 0.0)
    bc.create_value(my_model.function_space, T, t)

    # check that the value_fenics attribute is set correctly
    assert isinstance(bc.value_fenics, fem.Constant)

    # check the initial value of the boundary condition
    assert np.isclose(
        float(bc.value_fenics),
        float(value(t=float(t))),
    )

    # check the value of the boundary condition after updating the time
    for i in range(10):
        t.value = i
        bc.update(float(t))

        expected_value = float(value(t=float(t)))
        computed_value = float(bc.value_fenics)
        assert np.isclose(computed_value, expected_value)


def test_callable_x_only():
    """Test that the value attribute can be a callable function of x only"""

    # BUILD
    subdomain = F.SurfaceSubdomain1D(1, x=1)
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)

    def value(x):
        return 1.0 + x[0]

    species = F.Species("test")

    bc = F.DirichletBC(subdomain, value, species)

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[subdomain, vol_subdomain],
        species=[species],
    )

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()

    T = fem.Constant(my_model.mesh.mesh, 550.0)
    t = fem.Constant(my_model.mesh.mesh, 0.0)

    # TEST
    bc.create_value(my_model.function_space, T, t)

    # check that the value_fenics attribute is set correctly
    assert isinstance(bc.value_fenics, fem.Function)

    # check the initial value of the boundary condition
    assert np.isclose(
        bc.value_fenics.x.petsc_vec.array[-1],
        float(value(x=np.array([subdomain.x]))),
    )

    # check the value of the boundary condition after updating the time
    for i in range(10):
        t.value = i
        bc.update(float(t))
        expected_value = float(value(x=np.array([subdomain.x])))
        computed_value = bc.value_fenics.x.petsc_vec.array[-1]
        assert np.isclose(computed_value, expected_value)


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        lambda t: t,
        lambda t: 1.0 + t,
        lambda x: 1.0 + x[0],
        lambda x, t: 1.0 + x[0] + t,
        lambda x, t, T: 1.0 + x[0] + t + T,
        lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
        lambda t: 100.0 if t < 1 else 0.0,
    ],
)
def test_integration_with_HydrogenTransportProblem(value):
    """test that different callable functions can be applied to a dirichlet
    boundary condition, asserting in each case they match an expected value"""
    subdomain = F.SurfaceSubdomain1D(1, x=1)
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[vol_subdomain, subdomain],
    )
    my_model.species = [F.Species("H")]
    my_bc = F.DirichletBC(subdomain, value, "H")
    my_model.boundary_conditions = [my_bc]

    my_model.temperature = fem.Constant(my_model.mesh.mesh, 550.0)

    my_model.settings = F.Settings(atol=1, rtol=0.1, final_time=2)
    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    # RUN

    my_model.initialise()

    assert my_bc.value_fenics is not None

    my_model.run()

    # TEST

    if isinstance(value, float):
        expected_value = value
        computed_value = float(my_bc.value_fenics)
    elif callable(value):
        arguments = value.__code__.co_varnames
        if "x" in arguments and "t" in arguments and "T" in arguments:
            expected_value = value(x=np.array([subdomain.x]), t=2.0, T=550.0)
            computed_value = my_bc.value_fenics.x.petsc_vec.array[-1]
        elif "x" in arguments and "t" in arguments:
            expected_value = value(x=np.array([subdomain.x]), t=2.0)
            computed_value = my_bc.value_fenics.x.petsc_vec.array[-1]
        elif "x" in arguments:
            expected_value = value(x=np.array([subdomain.x]))
            computed_value = my_bc.value_fenics.x.petsc_vec.array[-1]
        elif "t" in arguments:
            expected_value = value(t=2.0)
            computed_value = float(my_bc.value_fenics)
        else:
            # test fails if lambda function is not recognised
            raise ValueError("value function not recognised")

    if isinstance(expected_value, Conditional):
        expected_value = float(expected_value)
    assert np.isclose(computed_value, expected_value)


@pytest.mark.parametrize(
    "value",
    [
        lambda t: ufl.conditional(ufl.lt(t, 1.0), 1, 2),
        lambda t: 1 + ufl.conditional(ufl.lt(t, 1.0), 1, 2.0),
        lambda t: 2 * ufl.conditional(ufl.lt(t, 1.0), 1, 2.0),
        lambda t: 2 / ufl.conditional(ufl.lt(t, 1.0), 1, 2.0),
    ],
)
def test_define_value_error_if_ufl_conditional_t_only(value):
    """Test that a ValueError is raised when the value attribute is a callable
    of t only and contains a ufl conditional"""

    subdomain = F.SurfaceSubdomain1D(1, x=1)
    species = F.Species("test")

    bc = F.DirichletBC(subdomain, value, species)

    t = fem.Constant(mesh, 0.0)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    with pytest.raises(
        ValueError, match="self.value should return a float or an int, not "
    ):
        bc.create_value(V, temperature=None, t=t)


def test_species_predefined():
    """Test a ValueError is raised when the species defined in the boundary
    condition is not predefined in the model"""

    subdomain = F.SurfaceSubdomain1D(1, x=1)
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[vol_subdomain, subdomain],
    )
    my_model.species = [F.Species("H")]
    my_bc = F.DirichletBC(subdomain, 1.0, "J")
    my_model.boundary_conditions = [my_bc]
    my_model.temperature = 1
    my_model.settings = F.Settings(atol=1, rtol=0.1)
    my_model.settings.stepsize = 1

    with pytest.raises(ValueError):
        my_model.initialise()


@pytest.mark.parametrize(
    "value_A, value_B",
    [
        (1.0, 1.0),
        (1.0, lambda t: t),
        (1.0, lambda t: 1.0 + t),
        (1.0, lambda x: 1.0 + x[0]),
        (1.0, lambda x, t: 1.0 + x[0] + t),
        (1.0, lambda x, t, T: 1.0 + x[0] + t + T),
        (1.0, lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0)),
    ],
)
def test_integration_with_a_multispecies_HydrogenTransportProblem(value_A, value_B):
    """test that a mixture of callable functions can be applied to dirichlet
    boundary conditions in a multispecies case, asserting in each case they
    match an expected value"""
    subdomain_A = F.SurfaceSubdomain1D(1, x=0)
    subdomain_B = F.SurfaceSubdomain1D(2, x=1)
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh),
        subdomains=[vol_subdomain, subdomain_A, subdomain_B],
    )
    my_model.species = [F.Species("A"), F.Species("B")]
    my_bc_A = F.DirichletBC(subdomain_A, value_A, "A")
    my_bc_B = F.DirichletBC(subdomain_B, value_B, "B")
    my_model.boundary_conditions = [my_bc_A, my_bc_B]

    my_model.temperature = fem.Constant(my_model.mesh.mesh, 550.0)

    my_model.settings = F.Settings(atol=1, rtol=0.1, final_time=2)
    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    # RUN

    my_model.initialise()

    for bc in [my_bc_A, my_bc_B]:
        assert bc.value_fenics is not None

    my_model.run()

    # TEST

    expected_value = value_A
    computed_value = float(my_bc_A.value_fenics)

    if isinstance(value_B, float):
        expected_value = value_B
        computed_value = float(my_bc_B.value_fenics)
    elif callable(value_B):
        arguments = value_B.__code__.co_varnames
        if "x" in arguments and "t" in arguments and "T" in arguments:
            expected_value = value_B(x=np.array([subdomain_B.x]), t=2.0, T=550.0)
            computed_value = my_bc_B.value_fenics.x.petsc_vec.array[-1]
        elif "x" in arguments and "t" in arguments:
            expected_value = value_B(x=np.array([subdomain_B.x]), t=2.0)
            computed_value = my_bc_B.value_fenics.x.petsc_vec.array[-1]
        elif "x" in arguments:
            expected_value = value_B(x=np.array([subdomain_B.x]))
            computed_value = my_bc_B.value_fenics.x.petsc_vec.array[-1]
        elif "t" in arguments:
            expected_value = value_B(t=2.0)
            computed_value = float(my_bc_B.value_fenics)
        else:
            # test fails if lambda function is not recognised
            raise ValueError("value function not recognised")

    if isinstance(expected_value, Conditional):
        expected_value = float(expected_value)
    assert np.isclose(computed_value, expected_value)


@pytest.mark.parametrize(
    "input, expected_value",
    [
        (1.0, False),
        (None, False),
        (fem.Constant(mesh, default_scalar_type(1.0)), False),
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
    subdomain = F.SurfaceSubdomain1D(1, x=0)
    species = F.Species("test")
    my_bc = F.DirichletBC(subdomain, input, species)

    assert my_bc.time_dependent is expected_value


@pytest.mark.parametrize(
    "input, expected_value",
    [
        (1.0, False),
        (None, False),
        (fem.Constant(mesh, default_scalar_type(1.0)), False),
        (lambda T: T, True),
        (lambda t: 1.0 + t, False),
        (lambda x, T: 1.0 + x[0] + T, True),
        (lambda x, t, T: 1.0 + x[0] + t + T, True),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
            False,
        ),
    ],
)
def test_bc_temperature_dependent_attribute(input, expected_value):
    """Test that the temperature_dependent attribute is correctly set"""
    subdomain = F.SurfaceSubdomain1D(1, x=0)
    species = F.Species("test")
    my_bc = F.DirichletBC(subdomain, input, species)

    assert my_bc.temperature_dependent is expected_value
