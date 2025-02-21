from mpi4py import MPI

import dolfinx.mesh
import pytest
import ufl
from dolfinx import fem

import festim as F

dummy_mat = F.Material(D_0=1, E_D=1, name="dummy_mat")

mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)


def test_init():
    """Test that the attributes are set correctly"""
    # create a Source object
    volume = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=dummy_mat)
    value = 1.0
    species = F.Species("test")
    source = F.ParticleSource(volume=volume, value=value, species=species)

    # check that the attributes are set correctly
    assert source.volume == volume
    assert source.species == species

    # check value is processed correctly
    assert source.value.input_value == value
    assert isinstance(source.value, F.Value)


@pytest.mark.parametrize(
    "value, expected_type",
    [
        (1.0, fem.Constant),
        (1, fem.Constant),
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
def test_create_fenics_object(value, expected_type):
    """Test that the correct fenics object is created depending on the value input"""

    # BUILD
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    species = F.Species("test")
    V = fem.functionspace(mesh, ("Lagrange", 1))

    source = F.ParticleSource(volume=vol_subdomain, value=value, species=species)

    T = fem.Constant(mesh, 550.0)
    t = fem.Constant(mesh, 0.0)

    # RUN
    source.value.convert_input_value(
        function_space=V, mesh=mesh, temperature=T, t=t, up_to_ufl_expr=True
    )

    # TEST
    # check that the value_fenics attribute is set correctly
    assert isinstance(source.value.fenics_object, expected_type)


@pytest.mark.parametrize(
    "input, expected_value",
    [
        (1.0, False),
        (None, False),
        (fem.Constant(mesh, dolfinx.default_scalar_type(1.0)), False),
        (lambda t: t, True),
        (lambda t: 1.0 + t, True),
        (lambda x: 1.0 + x[0], False),
        (lambda x, t: 1.0 + x[0] + t, True),
        (lambda x, t, T: 1.0 + x[0] + t + T, True),
        (lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0), True),
    ],
)
def test_source_time_dependent_attribute(input, expected_value):
    """Test that the time_dependent attribute is correctly set"""
    volume = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    species = F.Species("test")
    my_source = F.ParticleSource(input, volume, species)

    assert my_source.value.time_dependent is expected_value


@pytest.mark.parametrize(
    "input, expected_value",
    [
        (1.0, False),
        (None, False),
        (fem.Constant(mesh, dolfinx.default_scalar_type(1.0)), False),
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
def test_source_temperature_dependent_attribute(input, expected_value):
    """Test that the temperature_dependent attribute is correctly set"""
    volume = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    species = F.Species("test")
    my_source = F.ParticleSource(input, volume, species)

    assert my_source.value.temperature_dependent is expected_value


def test_ValueError_raised_when_callable_returns_wrong_type():
    """The create_value method should raise a ValueError when the callable
    returns an object which is not a float or int"""

    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)
    species = F.Species("test")

    def my_value(t):
        return ufl.conditional(ufl.lt(t, 0.5), 100, 0)

    source = F.ParticleSource(volume=vol_subdomain, value=my_value, species=species)

    T = fem.Constant(mesh, 550.0)
    t = fem.Constant(mesh, 0.0)

    with pytest.raises(
        ValueError,
        match="self.value should return a float or an int, not <class 'ufl.conditional.Conditional'",
    ):
        source.value.convert_input_value(
            mesh=mesh, temperature=T, t=t, up_to_ufl_expr=True
        )


def test_ValueError_raised_when_callable_returns_wrong_type_heat_source():
    """The create_value method should raise a ValueError when the callable
    returns an object which is not a float or int"""

    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)

    def my_value(t):
        return ufl.conditional(ufl.lt(t, 0.5), 100, 0)

    source = F.HeatSource(volume=vol_subdomain, value=my_value)

    t = fem.Constant(mesh, 0.0)

    with pytest.raises(
        ValueError,
        match="self.value should return a float or an int, not <class 'ufl.conditional.Conditional'",
    ):
        source.value.convert_input_value(mesh=mesh, t=t, up_to_ufl_expr=True)


@pytest.mark.parametrize(
    "volume_input",
    [
        1.0,
        "1",
        ["1"],
        [1.0],
        [[1]],
        [[F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)]],
        [[F.VolumeSubdomain(1, material=dummy_mat)]],
        None,
    ],
)
def test_TypeError_is_raised_when_volume_wrong_type(volume_input):
    """Test that a TypeError is raised when the volume is not of type
    festim.VolumeSubdomain"""

    my_spe = F.Species("test")
    with pytest.raises(
        TypeError,
        match="volume must be of type festim.VolumeSubdomain",
    ):
        F.ParticleSource(volume=volume_input, value=1.0, species=my_spe)


@pytest.mark.parametrize(
    "species_input",
    [
        1,
        1.0,
        [1],
        [1.0],
        [["test"]],
        [[F.Species("test")]],
        None,
    ],
)
def test_TypeError_is_raised_when_species_wrong_type(species_input):
    """Test that a TypeError is raised when the species is not of type
    festim.Species"""

    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=dummy_mat)

    with pytest.raises(
        TypeError,
        match="species must be of type festim.Species",
    ):
        F.ParticleSource(volume=my_vol, value=1.0, species=species_input)
