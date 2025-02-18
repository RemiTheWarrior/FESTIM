from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import pytest
import ufl
from dolfinx import fem

import festim as F


def henrys_law(T, H_0, E_H, pressure):
    """Applies the Henrys law to compute the concentration at the boundary"""
    H = H_0 * ufl.exp(-E_H / F.k_B / T)
    return H * pressure


def test_raise_error():
    """Test that a value error is raised if the pressure function is not supported in HenrysBC"""
    with pytest.raises(ValueError, match="pressure function not supported"):
        F.HenrysBC(subdomain=None, H_0=1.0, E_H=1.0, pressure=lambda c: c, species="H")


@pytest.mark.parametrize(
    "pressure",
    [
        100,
        lambda x: x,
        lambda t: t,
        lambda T: T,
        lambda x, t: x + 2 * t,
        lambda x, T: x + 2 * T,
        lambda t, T: 2 * t + 3 * T,
        lambda x, t, T: x + 2 * t + 3 * T,
        lambda t: 1 if t < 0.5 else 2,
    ],
)
def test_create_new_value_function(pressure):
    my_BC = F.HenrysBC(subdomain=None, H_0=1.0, E_H=1.0, pressure=pressure, species="H")
    assert my_BC.value is not None
    assert callable(my_BC.value)

    pressure_kwargs, value_kwargs = {}, {}
    if callable(pressure):
        if "x" in pressure.__code__.co_varnames:
            pressure_kwargs["x"] = 2.0
            value_kwargs["x"] = 2.0
        if "t" in pressure.__code__.co_varnames:
            pressure_kwargs["t"] = 3.0
            value_kwargs["t"] = 3.0
        if "T" in pressure.__code__.co_varnames:
            pressure_kwargs["T"] = 500.0
    value_kwargs["T"] = 500.0

    computed_value = my_BC.value(**value_kwargs)
    if callable(pressure):
        expected_value = henrys_law(
            T=500.0, H_0=1.0, E_H=1.0, pressure=pressure(**pressure_kwargs)
        )
    else:
        expected_value = henrys_law(T=500.0, H_0=1.0, E_H=1.0, pressure=pressure)
    assert np.isclose(computed_value, expected_value)


@pytest.mark.parametrize(
    "pressure",
    [
        100,
        lambda x: x[0],
        lambda t: t,
        lambda T: T,
        lambda x, t: x[0] + 2 * t,
        lambda x, T: x[0] + 2 * T,
        lambda t, T: 2 * t + 3 * T,
        lambda x, t, T: x[0] + 2 * t + 3 * T,
        lambda t: ufl.conditional(ufl.lt(t, 1.0), 100.0, 0.0),
        lambda t, x: ufl.conditional(ufl.lt(t, 1.0), 100.0 * x[0], 0.0),
    ],
)
def test_integration_with_HydrogenTransportProblem(pressure):
    subdomain = F.SurfaceSubdomain1D(1, x=1)
    dummy_mat = F.Material(1.0, 1.0, "dummy")
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 1], material=dummy_mat)

    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh(mesh), subdomains=[vol_subdomain, subdomain]
    )
    my_model.species = [F.Species("H")]
    my_bc = F.HenrysBC(
        subdomain=subdomain,
        H_0=2.0,
        E_H=0.5,
        pressure=pressure,
        species=my_model.species[0],
    )
    my_model.boundary_conditions = [my_bc]

    my_model.temperature = fem.Constant(my_model.mesh.mesh, 550.0)

    my_model.settings = F.Settings(atol=1, rtol=0.1, final_time=2)
    my_model.settings.stepsize = F.Stepsize(initial_value=1)

    # RUN

    my_model.initialise()

    assert my_bc.value_fenics is not None

    my_model.run()
