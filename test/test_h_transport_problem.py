import mpi4py.MPI as MPI

import dolfinx.mesh
import numpy as np
import pytest
import tqdm.autonotebook
import ufl
from dolfinx import default_scalar_type, fem, nls

import festim as F

test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
x = ufl.SpatialCoordinate(test_mesh.mesh)
dummy_mat = F.Material(D_0=1, E_D=1, name="dummy_mat")


# TODO test all the methods in the class
@pytest.mark.parametrize(
    "value",
    [
        1,
        fem.Constant(test_mesh.mesh, default_scalar_type(1.0)),
        1.0,
        "coucou",
        lambda x: 2 * x[0],
    ],
)
def test_temperature_setter_type(value):
    """Test that the temperature type is correctly set"""
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
    )

    if not isinstance(value, (fem.Constant, int, float)):
        if callable(value):
            my_model.temperature = value
        else:
            with pytest.raises(TypeError):
                my_model.temperature = value


@pytest.mark.parametrize(
    "input, expected_value",
    [
        (1.0, False),
        (1, False),
        (lambda t: t, True),
        (lambda t: 1.0 + t, True),
        (lambda x: 1.0 + x[0], False),
        (lambda x, t: 1.0 + x[0] + t, True),
        (lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0), True),
    ],
)
def test_time_dependent_temperature_attribute(input, expected_value):
    """Test that the temperature_time_dependent attribute is correctly set"""

    my_model = F.HydrogenTransportProblem()
    my_model.temperature = input

    assert my_model.temperature_time_dependent is expected_value


def test_define_temperature_value_error_raised():
    """Test that a ValueError is raised when the temperature is None"""

    # BUILD
    my_model = F.HydrogenTransportProblem(mesh=test_mesh)

    my_model.temperature = None

    # TEST
    with pytest.raises(
        ValueError, match="the temperature attribute needs to be defined"
    ):
        my_model.define_temperature()


@pytest.mark.parametrize(
    "input, expected_type",
    [
        (1.0, fem.Constant),
        (1, fem.Constant),
        (fem.Constant(test_mesh.mesh, default_scalar_type(1.0)), fem.Constant),
        (lambda t: t, fem.Constant),
        (lambda t: 1.0 + t, fem.Constant),
        (lambda x: 1.0 + x[0], fem.Function),
        (lambda x, t: 1.0 + x[0] + t, fem.Function),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.0), 100.0 + x[0], 0.0),
            fem.Function,
        ),
        (lambda t: 100.0 if t < 1 else 0.0, fem.Constant),
    ],
)
def test_define_temperature(input, expected_type):
    """Test that the define_temperature method correctly sets the
    temperature_fenics attribute to either a fem.Constant or a
    fem.Function depending on the type of input"""

    # BUILD
    my_model = F.HydrogenTransportProblem(mesh=test_mesh)
    my_model.t = fem.Constant(test_mesh.mesh, 0.0)

    my_model.temperature = input

    # RUN
    my_model.define_temperature()

    # TEST
    assert isinstance(my_model.temperature_fenics, expected_type)


@pytest.mark.parametrize(
    "input",
    [
        lambda t: ufl.conditional(ufl.lt(t, 1.0), 1, 2),
        lambda t: 1 + ufl.conditional(ufl.lt(t, 1.0), 1, 2.0),
        lambda t: 2 * ufl.conditional(ufl.lt(t, 1.0), 1, 2.0),
        lambda t: 2 / ufl.conditional(ufl.lt(t, 1.0), 1, 2.0),
    ],
)
def test_define_temperature_error_if_ufl_conditional_t_only(input):
    """Test that a ValueError is raised when the temperature attribute is a callable
    of t only and contains a ufl conditional"""
    my_model = F.HydrogenTransportProblem(mesh=test_mesh)
    my_model.t = fem.Constant(test_mesh.mesh, 0.0)

    my_model.temperature = input

    with pytest.raises(
        ValueError,
        match="self.temperature should return a float or an int, not",
    ):
        my_model.define_temperature()


def test_iterate():
    """Test that the iterate method updates the solution and time correctly"""
    # BUILD
    my_model = F.HydrogenTransportProblem()

    my_model.settings = F.Settings(atol=1e-6, rtol=1e-6, final_time=10)
    my_model.settings.stepsize = 2.0

    my_model.progress_bar = tqdm.autonotebook.tqdm(
        desc="Solving H transport problem",
        total=my_model.settings.final_time,
        unit_scale=True,
    )

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    V = fem.functionspace(mesh, ("Lagrange", 1))
    my_model.u = fem.Function(V)
    my_model.u_n = fem.Function(V)
    my_model.dt = fem.Constant(mesh, 2.0)
    v = ufl.TestFunction(V)

    source_value = 2.0
    form = (
        my_model.u - my_model.u_n
    ) / my_model.dt * v * ufl.dx - source_value * v * ufl.dx

    problem = fem.petsc.NonlinearProblem(form, my_model.u, bcs=[])
    my_model.solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

    my_model.t = fem.Constant(mesh, 0.0)

    for i in range(10):
        # RUN
        my_model.iterate()

        # TEST

        # check that t evolves
        expected_t_value = (i + 1) * float(my_model.dt)
        assert np.isclose(float(my_model.t), expected_t_value)

        # check that u and u_n are updated
        expected_u_value = (i + 1) * float(my_model.dt) * source_value
        assert np.all(np.isclose(my_model.u.x.array, expected_u_value))


@pytest.mark.parametrize(
    "T_function, expected_values",
    [
        (lambda t: t, [1.0, 2.0, 3.0]),
        (lambda t: 1.0 + t, [2.0, 3.0, 4.0]),
        (lambda x, t: 1.0 + x[0] + t, [6.0, 7.0, 8.0]),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.5), 100.0 + x[0], 0.0),
            [104.0, 0.0, 0.0],
        ),
    ],
)
def test_update_time_dependent_values_temperature(T_function, expected_values):
    """Test that different time-dependent callable functions for the
    temperature are updated at each time step and match an expected value"""

    # BUILD
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
    )
    my_model.t = fem.Constant(my_model.mesh.mesh, 0.0)
    dt = fem.Constant(test_mesh.mesh, 1.0)

    my_model.temperature = T_function

    my_model.define_temperature()

    for i in range(3):
        # RUN
        my_model.t.value += dt.value
        my_model.update_time_dependent_values()

        # TEST
        if isinstance(my_model.temperature_fenics, fem.Constant):
            computed_value = float(my_model.temperature_fenics)
            assert np.isclose(computed_value, expected_values[i])


def test_initialise_exports_find_species_with_one_field():
    """Test that a species can be found from the model species if given as a string"""

    # BUILD
    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0])),
        species=[F.Species("H"), F.Species("D")],
        temperature=1,
    )
    surf = F.SurfaceSubdomain1D(id=1, x=1)

    my_model.exports = [F.SurfaceFlux(field="J", surface=surf)]
    my_model.define_function_spaces()

    # TEST
    with pytest.raises(ValueError, match="Species J not found in list of species"):
        my_model.initialise_exports()


def test_define_D_global_different_temperatures():
    """Test that the D_global object is correctly defined when the temperature
    is different in the volume subdomains"""
    D_0, E_D = 1.5, 0.1
    my_mat = F.Material(D_0=D_0, E_D=E_D, name="my_mat")
    surf = F.SurfaceSubdomain1D(id=1, x=0)
    H = F.Species("H")

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0, 4, num=101)),
        subdomains=[
            F.VolumeSubdomain1D(id=1, borders=[0, 2], material=my_mat),
            F.VolumeSubdomain1D(id=2, borders=[2, 4], material=my_mat),
        ],
        species=[H],
        temperature=lambda x: 100.0 * x[0] + 50,
    )

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()
    my_model.t = 1
    my_model.define_temperature()

    D_computed, D_expr = my_model.define_D_global(H)

    computed_values = [D_computed.x.array[0], D_computed.x.array[-1]]

    D_analytical_left = D_0 * np.exp(-E_D / (F.k_B * 50))
    D_analytical_right = D_0 * np.exp(-E_D / (F.k_B * 450))

    expected_values = [D_analytical_left, D_analytical_right]

    assert np.isclose(computed_values, expected_values).all()


def test_define_D_global_different_materials():
    """Test that the D_global object is correctly defined when the material
    is different in the volume subdomains"""
    D_0_left, E_D_left = 1.0, 0.1
    D_0_right, E_D_right = 2.0, 0.2
    my_mat_L = F.Material(D_0=D_0_left, E_D=E_D_left, name="my_mat_L")
    my_mat_R = F.Material(D_0=D_0_right, E_D=E_D_right, name="my_mat_R")
    H = F.Species("H")

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0, 4, num=101)),
        subdomains=[
            F.VolumeSubdomain1D(id=1, borders=[0, 2], material=my_mat_L),
            F.VolumeSubdomain1D(id=2, borders=[2, 4], material=my_mat_R),
        ],
        species=[H],
        temperature=500,
    )

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()
    my_model.t = 0
    my_model.define_temperature()

    D_computed, D_expr = my_model.define_D_global(H)

    computed_values = [D_computed.x.array[0], D_computed.x.array[-1]]

    D_expected_left = D_0_left * np.exp(-E_D_left / (F.k_B * my_model.temperature))
    D_expected_right = D_0_right * np.exp(-E_D_right / (F.k_B * my_model.temperature))

    expected_values = [D_expected_left, D_expected_right]

    assert np.isclose(computed_values, expected_values).all()


def test_initialise_exports_multiple_exports_same_species():
    """Test that the diffusion coefficient within the D_global object function is the same
    for multiple exports of the same species, and that D_global object is only
    created once per species"""

    D_0, E_D = 1.5, 0.1
    my_mat = F.Material(D_0=D_0, E_D=E_D, name="my_mat")
    surf_1 = F.SurfaceSubdomain1D(id=1, x=0)
    surf_2 = F.SurfaceSubdomain1D(id=1, x=4)
    H = F.Species("H")

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0, 4, num=101)),
        subdomains=[
            F.VolumeSubdomain1D(id=1, borders=[0, 2], material=my_mat),
            F.VolumeSubdomain1D(id=2, borders=[2, 4], material=my_mat),
        ],
        species=[H],
        temperature=500,
        exports=[
            F.SurfaceFlux(
                field=H,
                surface=surf_1,
            ),
            F.SurfaceFlux(
                field=H,
                surface=surf_2,
            ),
        ],
    )

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()
    my_model.t = 0
    my_model.define_temperature()
    my_model.initialise_exports()

    Ds = [export.D for export in my_model.exports]

    assert Ds[0].x.array[0] == Ds[1].x.array[0]


def test_export_resets_quantities():
    """Test that the export.data and export.t are correctly reset every time a simulation is initiated."""
    my_mat = F.Material(D_0=1, E_D=0)
    H = F.Species("H")
    surf = F.SurfaceSubdomain1D(id=1, x=4)

    my_export = F.SurfaceFlux(
        field=H,
        surface=surf,
    )

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D([0, 1, 2, 3, 4]),
        subdomains=[F.VolumeSubdomain1D(id=1, borders=[0, 4], material=my_mat), surf],
        species=[H],
        temperature=400,
        exports=[my_export],
    )

    my_model.settings = F.Settings(atol=1e-15, rtol=1e-15, transient=False)

    for i in range(3):
        my_model.initialise()
        my_model.run()

        assert my_model.exports[0].t == [0.0]
        assert my_model.exports[0].data == [0.0]


def test_define_D_global_multispecies():
    """Test that the D_global object is correctly defined when there are multiple
    species in one subdomain"""
    A = F.Species("A")
    B = F.Species("B")

    D_0_A, D_0_B = 1.0, 2.0
    E_D_A, E_D_B = 0.1, 0.2

    my_mat = F.Material(
        D_0={A: D_0_A, B: D_0_B}, E_D={A: E_D_A, B: E_D_B}, name="my_mat"
    )
    surf = F.SurfaceSubdomain1D(id=1, x=1)

    my_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0, 1, num=101)),
        subdomains=[
            F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat),
        ],
        species=[F.Species("A"), F.Species("B")],
        temperature=500,
    )

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()
    my_model.t = 0
    my_model.define_temperature()

    D_A_computed, D_A_expr = my_model.define_D_global(A)
    D_B_computed, D_B_expr = my_model.define_D_global(B)

    computed_values = [D_A_computed.x.array[-1], D_B_computed.x.array[-1]]

    D_expected_A = D_0_A * np.exp(-E_D_A / (F.k_B * my_model.temperature))
    D_expected_B = D_0_B * np.exp(-E_D_B / (F.k_B * my_model.temperature))

    expected_values = [D_expected_A, D_expected_B]

    assert np.isclose(computed_values, expected_values).all()


def test_post_processing_update_D_global():
    """Test that the D_global object is updated at each time
    step when temperture is time dependent"""
    my_mesh = F.Mesh1D(np.linspace(0, 1, num=11))
    my_mat = F.Material(D_0=1.5, E_D=0.1, name="my_mat")
    surf = F.SurfaceSubdomain1D(id=1, x=1)

    # create species and interpolate solution
    H = F.Species("H")
    V = fem.functionspace(my_mesh.mesh, ("Lagrange", 1))
    u = fem.Function(V)
    u.interpolate(lambda x: 2 * x[0] ** 2 + 1)
    H.solution = u

    my_export = F.SurfaceFlux(
        field=H,
        surface=surf,
    )

    # Build the model
    my_model = F.HydrogenTransportProblem(
        mesh=my_mesh,
        subdomains=[
            F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat),
            surf,
        ],
        species=[H],
        temperature=lambda t: 500 * t,
        exports=[my_export],
    )

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()
    my_model.t = fem.Constant(my_model.mesh.mesh, 1.0)
    my_model.define_temperature()
    my_model.initialise_exports()

    # RUN
    my_model.post_processing()
    value_t_1 = my_export.D.x.array[-1]

    my_model.t = fem.Constant(my_model.mesh.mesh, 2.0)
    my_model.update_time_dependent_values()
    my_model.post_processing()
    value_t_2 = my_export.D.x.array[-1]

    # TEST
    assert value_t_1 != value_t_2


def test_post_processing_update_D_global_2():
    """Test that the D_global object is updated at each time
    step when temperture is time dependent"""
    my_mesh = F.Mesh1D(np.linspace(0, 1, num=11))
    my_mat = F.Material(D_0=1.5, E_D=0.1, name="my_mat")
    surf = F.SurfaceSubdomain1D(id=1, x=1)

    # create species and interpolate solution
    H = F.Species("H")
    V = fem.functionspace(my_mesh.mesh, ("Lagrange", 1))
    u = fem.Function(V)
    u.interpolate(lambda x: x[0] ** 2 + 100)
    H.solution = u

    my_export = F.MaximumSurface(
        field=H,
        surface=surf,
    )

    # Build the model
    my_model = F.HydrogenTransportProblem(
        mesh=my_mesh,
        subdomains=[
            F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat),
            surf,
        ],
        species=[H],
        temperature=lambda t: 500 * t,
        exports=[my_export],
    )

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()
    my_model.t = fem.Constant(my_model.mesh.mesh, 1.0)
    my_model.define_temperature()
    my_model.initialise_exports()

    # RUN
    my_model.post_processing()
    my_value = my_export.D.x.array[-1]
    assert isinstance(my_value, float)


def test_post_processing_update_D_global_volume_1():
    """Test that the D_global object is updated at each time
    step when temperture is time dependent"""
    my_mesh = F.Mesh1D(np.linspace(0, 1, num=11))
    my_mat = F.Material(D_0=1.5, E_D=0.1, name="my_mat")
    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)

    # create species and interpolate solution
    H = F.Species("H")
    V = fem.functionspace(my_mesh.mesh, ("Lagrange", 1))
    u = fem.Function(V)
    u.interpolate(lambda x: x[0] ** 2 + 100)
    H.solution = u

    my_export = F.AverageVolume(field=H, volume=my_vol)

    # Build the model
    my_model = F.HydrogenTransportProblem(
        mesh=my_mesh,
        subdomains=[my_vol],
        species=[H],
        temperature=lambda t: 500 * t,
        exports=[my_export],
    )

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()
    my_model.t = fem.Constant(my_model.mesh.mesh, 1.0)
    my_model.define_temperature()
    my_model.initialise_exports()

    # RUN
    my_model.post_processing()
    my_value = my_model.exports[0].data[-1]
    assert isinstance(my_value, float)


def test_post_processing_update_D_global_volume_2():
    """Test that the D_global object is updated at each time
    step when temperture is time dependent"""
    my_mesh = F.Mesh1D(np.linspace(0, 1, num=11))
    my_mat = F.Material(D_0=1.5, E_D=0.1, name="my_mat")
    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)

    # create species and interpolate solution
    H = F.Species("H")
    V = fem.functionspace(my_mesh.mesh, ("Lagrange", 1))
    u = fem.Function(V)
    u.interpolate(lambda x: x[0] ** 2 + 100)
    H.solution = u

    my_export = F.MaximumVolume(field=H, volume=my_vol)

    # Build the model
    my_model = F.HydrogenTransportProblem(
        mesh=my_mesh,
        subdomains=[my_vol],
        species=[H],
        temperature=lambda t: 500 * t,
        exports=[my_export],
    )

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()
    my_model.t = fem.Constant(my_model.mesh.mesh, 1.0)
    my_model.define_temperature()
    my_model.initialise_exports()

    # RUN
    my_model.post_processing()
    my_value = my_model.exports[0].data[-1]
    assert isinstance(my_value, float)


@pytest.mark.parametrize(
    "temperature_value, bc_value, expected_values",
    [
        (5, 1.0, [1.0, 1.0, 1.0]),
        (lambda t: t + 1, lambda T: T, [2.0, 3.0, 4.0]),
        (lambda x: 1 + x[0], lambda T: 2 * T, [10.0, 10.0, 10.0]),
        (lambda x, t: t + x[0], lambda T: 0.5 * T, [2.5, 3.0, 3.5]),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 2), 3 + x[0], 0.0),
            lambda T: T,
            [7.0, 0.0, 0.0],
        ),
    ],
)
def test_update_time_dependent_bcs_with_time_dependent_temperature(
    temperature_value, bc_value, expected_values
):
    """Test that temperature dependent bcs are updated at each time step when the
    temperature is time dependent, and match an expected value"""

    # BUILD
    H = F.Species("H")
    volume_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    surface_subdomain = F.SurfaceSubdomain1D(id=1, x=4)
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        species=[H],
        subdomains=[volume_subdomain, surface_subdomain],
    )
    my_model.t = fem.Constant(my_model.mesh.mesh, 0.0)
    dt = fem.Constant(test_mesh.mesh, 1.0)

    my_model.temperature = temperature_value
    my_bc = F.DirichletBC(species=H, value=bc_value, subdomain=surface_subdomain)
    my_model.boundary_conditions = [my_bc]

    my_model.define_temperature()
    my_model.define_function_spaces()
    my_model.assign_functions_to_species()
    my_model.define_meshtags_and_measures()
    my_model.create_dirichletbc_form(my_bc)

    for i in range(3):
        # RUN
        my_model.t.value += dt.value
        my_model.update_time_dependent_values()

        # TEST
        if isinstance(my_model.boundary_conditions[0].value_fenics, fem.Constant):
            computed_value = float(my_model.boundary_conditions[0].value_fenics)
            assert np.isclose(computed_value, expected_values[i])


@pytest.mark.parametrize(
    "source_value, expected_values",
    [
        (lambda t: t, [1.0, 2.0, 3.0]),
        (lambda t: 1.0 + t, [2.0, 3.0, 4.0]),
        (lambda x, t: 1.0 + x[0] + t, [6.0, 7.0, 8.0]),
        (lambda T, t: T + 2 * t, [12.0, 14.0, 16.0]),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.5), 100.0 + x[0], 0.0),
            [104.0, 0.0, 0.0],
        ),
    ],
)
def test_update_time_dependent_values_source(source_value, expected_values):
    """Test that time dependent sources are updated at each time step,
    and match an expected value"""
    # BUILD
    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    H = F.Species("H")
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh, temperature=10, subdomains=[my_vol], species=[H]
    )
    my_model.t = fem.Constant(my_model.mesh.mesh, 0.0)
    dt = fem.Constant(test_mesh.mesh, 1.0)

    my_source = F.ParticleSource(value=source_value, volume=my_vol, species=H)
    my_model.sources = [my_source]

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()
    my_model.assign_functions_to_species()
    my_model.define_temperature()
    my_model.create_source_values_fenics()

    for i in range(3):
        # RUN
        my_model.t.value += dt.value
        my_model.update_time_dependent_values()

        # TEST
        if isinstance(my_model.sources[0].value.fenics_object, fem.Constant):
            computed_value = float(my_model.sources[0].value.fenics_object)
            assert np.isclose(computed_value, expected_values[i])


@pytest.mark.parametrize(
    "temperature_value, source_value, expected_values",
    [
        (5, 1.0, [1.0, 1.0, 1.0]),
        (lambda t: t + 1, lambda T: T, [2.0, 3.0, 4.0]),
        (lambda x: 1 + x[0], lambda T: 2 * T, [10.0, 10.0, 10.0]),
        (lambda x, t: t + x[0], lambda T: 0.5 * T, [2.5, 3.0, 3.5]),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 2), 3 + x[0], 0.0),
            lambda T: T,
            [7.0, 0.0, 0.0],
        ),
    ],
)
def test_update_sources_with_time_dependent_temperature(
    temperature_value, source_value, expected_values
):
    """Test that temperature dependent source terms are updated at each time step
    when the temperature is time dependent, and match an expected value"""

    # BUILD
    H = F.Species("H")
    volume_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    surface_subdomain = F.SurfaceSubdomain1D(id=1, x=4)
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        species=[H],
        subdomains=[volume_subdomain, surface_subdomain],
        temperature=temperature_value,
    )
    my_model.t = fem.Constant(my_model.mesh.mesh, 0.0)
    dt = fem.Constant(test_mesh.mesh, 1.0)

    my_model.sources = [
        F.ParticleSource(value=source_value, volume=volume_subdomain, species=H)
    ]

    my_model.define_temperature()
    my_model.define_function_spaces()
    my_model.assign_functions_to_species()
    my_model.define_meshtags_and_measures()
    my_model.create_source_values_fenics()

    for i in range(3):
        # RUN
        my_model.t.value += dt.value
        my_model.update_time_dependent_values()

        # TEST
        if isinstance(my_model.sources[0].value.fenics_object, fem.Constant):
            computed_value = float(my_model.sources[0].value.fenics_object)
            assert np.isclose(computed_value, expected_values[i])


def test_create_source_values_fenics_multispecies():
    """Test that the define_sources method correctly sets the value_fenics attribute in
    a multispecies case"""
    # BUILD
    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    H, D = F.Species("H"), F.Species("D")
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        temperature=10,
        subdomains=[my_vol],
        species=[H, D],
    )
    my_model.t = fem.Constant(my_model.mesh.mesh, 4.0)

    my_source_1 = F.ParticleSource(value=lambda t: t + 1, volume=my_vol, species=H)
    my_source_2 = F.ParticleSource(value=lambda t: 2 * t + 3, volume=my_vol, species=D)
    my_model.sources = [my_source_1, my_source_2]

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()
    my_model.assign_functions_to_species()
    my_model.define_temperature()

    # RUN
    my_model.create_source_values_fenics()

    # TEST
    assert np.isclose(float(my_model.sources[0].value.fenics_object), 5)
    assert np.isclose(float(my_model.sources[1].value.fenics_object), 11)


# TODO replace this by a proper MMS test
def test_run_in_steady_state():
    """Test that the run method works in steady state"""
    # BUILD
    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        temperature=500,
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
        subdomains=[my_vol],
        species=[F.Species("H")],
    )

    my_model.initialise()

    # RUN
    my_model.run()

    # TEST
    assert my_model.t.value == 0.0


def test_species_setter():
    """Test that a TypeError is rasied when a species of type other than F.Species is
    given"""

    my_model = F.HydrogenTransportProblem()

    with pytest.raises(
        TypeError,
        match="elements of species must be of type festim.Species not <class 'int'>",
    ):
        my_model.species = [1, 2, 3]


def test_create_initial_conditions_ValueError_raised_when_not_transient():
    """Test that ValueError is raised if initial conditions are defined in
    a steady state simulation"""

    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    H = F.Species("H")
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        temperature=10,
        subdomains=[my_vol],
        species=[H],
        initial_conditions=[F.InitialCondition(value=1.0, species=H)],
        settings=F.Settings(atol=1, rtol=1, transient=False),
    )

    with pytest.raises(
        ValueError,
        match="Initial conditions can only be defined for transient simulations",
    ):
        my_model.initialise()


@pytest.mark.parametrize(
    "input_value, expected_value",
    [
        (1.0, 1.0),
        (1, 1.0),
        (lambda T: 1.0 + T, 11.0),
        (lambda x: 1.0 + x[0], 5.0),
        (lambda x, T: 1.0 + x[0] + T, 15.0),
    ],
)
def test_create_initial_conditions_expr_fenics(input_value, expected_value):
    """Test that after calling create_initial_conditions, the prev_solution
    attribute of the species has the correct value at x=4.0."""

    # BUILD
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 4], material=dummy_mat)
    H = F.Species("H")
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        temperature=10,
        subdomains=[vol_subdomain],
        species=[H],
        initial_conditions=[F.InitialCondition(value=input_value, species=H)],
        settings=F.Settings(atol=1, rtol=1, final_time=2, stepsize=1),
    )

    # RUN
    my_model.initialise()

    assert np.isclose(
        my_model.species[0].prev_solution.x.petsc_vec.array[-1],
        expected_value,
    )


def test_create_species_from_trap():
    "Test that a new species and reaction is created when a trap is given"

    # BUILD
    my_model = F.HydrogenTransportProblem(mesh=test_mesh)
    my_mobile_species = F.Species("test_mobile")
    mat = F.Material(D_0=1, E_D=1, name="mat")
    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=mat)
    my_trap = F.Trap(
        name="test_trap",
        mobile_species=my_mobile_species,
        k_0=1,
        E_k=1,
        p_0=1,
        E_p=1,
        n=1,
        volume=my_vol,
    )
    my_settings = F.Settings(atol=1, rtol=1, transient=False)
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        subdomains=[my_vol],
        species=[my_mobile_species],
        traps=[my_trap],
        temperature=100,
        settings=my_settings,
    )

    # RUN
    my_model.initialise()

    # TEST
    # test that an additional species is generated
    assert len(my_model.species) == 2
    assert isinstance(my_model.species[1], F.Species)

    assert len(my_model.reactions) == 1
    assert isinstance(my_model.reactions[0], F.Reaction)


@pytest.mark.parametrize(
    "input_value_1, input_value_2, expected_value_1, expected_value_2",
    [
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 1, 1.0, 1.0),
        (1.0, lambda T: 1.0 + T, 1.0, 11.0),
        (1.0, lambda x: 1.0 + x[0], 1.0, 5.0),
        (1.0, lambda x, T: 1.0 + x[0] + T, 1.0, 15.0),
    ],
)
def test_create_initial_conditions_value_fenics_multispecies(
    input_value_1, input_value_2, expected_value_1, expected_value_2
):
    """Test that after calling create_initial_conditions, the prev_solution
    attribute of each species has the correct value at x=4.0 in a multispecies case"""

    # BUILD
    test_mesh = F.Mesh1D(vertices=np.linspace(0, 4, num=101))
    vol_subdomain = F.VolumeSubdomain1D(1, borders=[0, 4], material=dummy_mat)
    H, D = F.Species("H"), F.Species("D")
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        temperature=10,
        subdomains=[vol_subdomain],
        species=[H, D],
        initial_conditions=[
            F.InitialCondition(value=input_value_2, species=D),
            F.InitialCondition(value=input_value_1, species=H),
        ],
        settings=F.Settings(atol=1, rtol=1, final_time=2, stepsize=1),
    )

    # RUN
    my_model.initialise()

    # TEST
    # When in multispecies, the u and u_n x arrays are structured as follows:
    # [H, D, ..., H, D, H, D], thus the last two values are the ones we are
    # interested in

    # test value of H at x = 4.0
    assert np.isclose(my_model.u_n.x.array[-2], expected_value_1)

    # test value of D at x = 4.0
    assert np.isclose(my_model.u_n.x.array[-1], expected_value_2)


def test_adaptive_timestepping_grows():
    """Tests that the stepsize grows"""
    # BUILD
    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        temperature=500,
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=True, final_time=10),
        subdomains=[my_vol],
        species=[F.Species("H")],
    )

    stepsize = F.Stepsize(initial_value=1)
    stepsize.growth_factor = 1.2
    stepsize.target_nb_iterations = 100  # force it to always grow
    my_model.settings.stepsize = stepsize

    my_model.initialise()

    my_model.progress_bar = tqdm.autonotebook.tqdm(
        desc="Solving H transport problem",
        total=my_model.settings.final_time,
        unit_scale=True,
    )

    # RUN & TEST
    previous_value = stepsize.initial_value
    while my_model.t.value < my_model.settings.final_time:
        my_model.iterate()

        # check that the current value is greater than the previous one
        assert my_model.dt.value > previous_value

        previous_value = float(my_model.dt)


def test_adaptive_timestepping_shrinks():
    """Tests that the stepsize shrinks"""
    # BUILD
    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        temperature=500,
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=True, final_time=10),
        subdomains=[my_vol],
        species=[F.Species("H")],
    )

    stepsize = F.Stepsize(initial_value=1)
    stepsize.cutback_factor = 0.8
    stepsize.target_nb_iterations = -1  # force it to always shrink
    my_model.settings.stepsize = stepsize

    my_model.initialise()

    my_model.progress_bar = tqdm.autonotebook.tqdm(
        desc="Solving H transport problem",
        total=my_model.settings.final_time,
        unit_scale=True,
    )

    # RUN & TEST
    previous_value = stepsize.initial_value
    while my_model.t.value < my_model.settings.final_time and my_model.dt.value > 0.1:
        my_model.iterate()

        # check that the current value is smaller than the previous one
        assert my_model.dt.value < previous_value

        previous_value = float(my_model.dt)


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("species", F.Species("test")),
        ("reactions", None),
        ("sources", None),
        ("subdomains", None),
        ("boundary_conditions", None),
        ("exports", None),
    ],
)
def test_reinstantiation_of_class(attribute, value):
    """Test that when an attribute defaults to empty list, when the class
    is reinstantiated the list is not passed to the new class object"""

    model_1 = F.HydrogenTransportProblem()
    getattr(model_1, attribute).append(value)

    model_2 = F.HydrogenTransportProblem()
    assert len(getattr(model_2, attribute)) == 0


def test_define_meshtags_and_measures_with_custom_fenics_mesh():
    """Test that the define_meshtags_and_measures method works when the mesh is
    a custom fenics mesh"""

    # BUILD
    mesh_1D = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    # 1D meshtags
    my_surface_meshtags = dolfinx.mesh.meshtags(
        mesh_1D,
        0,
        np.array([0, 10], dtype=np.int32),
        np.array([1, 2], dtype=np.int32),
    )

    num_cells = mesh_1D.topology.index_map(1).size_local
    my_volume_meshtags = dolfinx.mesh.meshtags(
        mesh_1D,
        1,
        np.arange(num_cells, dtype=np.int32),
        np.full(num_cells, 1, dtype=np.int32),
    )

    my_mesh = F.Mesh(mesh=mesh_1D)

    my_model = F.HydrogenTransportProblem(mesh=my_mesh)
    my_model.facet_meshtags = my_surface_meshtags
    my_model.volume_meshtags = my_volume_meshtags

    # TEST
    my_model.define_meshtags_and_measures()


def test_error_raised_when_custom_fenics_mesh_wrong_facet_meshtags_type():
    """Test the facet_meshtags type hinting raises error when given as wrong type"""

    # BUILD
    mesh_1D = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    my_mesh = F.Mesh(mesh=mesh_1D)
    my_model = F.HydrogenTransportProblem(mesh=my_mesh)

    # TEST
    with pytest.raises(TypeError, match="value must be of type dolfinx.mesh.MeshTags"):
        my_model.facet_meshtags = [0, 1]


def test_error_raised_when_custom_fenics_mesh_wrong_volume_meshtags_type():
    """Test the volume_meshtags type hinting raises error when given as wrong type"""

    # BUILD
    mesh_1D = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    my_mesh = F.Mesh(mesh=mesh_1D)
    my_model = F.HydrogenTransportProblem(mesh=my_mesh)

    # TEST
    with pytest.raises(TypeError, match="value must be of type dolfinx.mesh.MeshTags"):
        my_model.volume_meshtags = [0, 1]


@pytest.mark.parametrize(
    "bc_value, expected_values",
    [
        (lambda t: t, [1.0, 2.0, 3.0]),
        (lambda t: 1.0 + t, [2.0, 3.0, 4.0]),
        (lambda x, t: 1.0 + x[0] + t, [6.0, 7.0, 8.0]),
        (lambda T, t: T + 2 * t, [12.0, 14.0, 16.0]),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 1.5), 100.0 + x[0], 0.0),
            [104.0, 0.0, 0.0],
        ),
    ],
)
def test_update_time_dependent_values_flux(bc_value, expected_values):
    """Test that time dependent fluxes are updated at each time step,
    and match an expected value"""
    # BUILD
    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    surface = F.SurfaceSubdomain1D(id=2, x=0)
    H = F.Species("H")
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        temperature=10,
        subdomains=[my_vol, surface],
        species=[H],
    )
    my_model.t = fem.Constant(my_model.mesh.mesh, 0.0)
    dt = fem.Constant(test_mesh.mesh, 1.0)

    my_bc = F.ParticleFluxBC(subdomain=surface, value=bc_value, species=H)
    my_model.boundary_conditions = [my_bc]

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()
    my_model.assign_functions_to_species()
    my_model.define_temperature()
    my_model.define_boundary_conditions()

    for i in range(3):
        # RUN
        my_model.t.value += dt.value
        my_model.update_time_dependent_values()

        # TEST
        if isinstance(my_model.boundary_conditions[0].value_fenics, fem.Constant):
            computed_value = float(my_model.boundary_conditions[0].value_fenics)
            assert np.isclose(computed_value, expected_values[i])


@pytest.mark.parametrize(
    "temperature_value, bc_value, expected_values",
    [
        (5, 1.0, [1.0, 1.0, 1.0]),
        (lambda t: t + 1, lambda T: T, [2.0, 3.0, 4.0]),
        (lambda x: 1 + x[0], lambda T: 2 * T, [10.0, 10.0, 10.0]),
        (lambda x, t: t + x[0], lambda T: 0.5 * T, [2.5, 3.0, 3.5]),
        (
            lambda x, t: ufl.conditional(ufl.lt(t, 2), 3 + x[0], 0.0),
            lambda T: T,
            [7.0, 0.0, 0.0],
        ),
    ],
)
def test_update_fluxes_with_time_dependent_temperature(
    temperature_value, bc_value, expected_values
):
    """Test that temperature dependent flux terms are updated at each time step
    when the temperature is time dependent, and match an expected value"""

    # BUILD
    H = F.Species("H")
    volume_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    surface_subdomain = F.SurfaceSubdomain1D(id=1, x=4)
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        species=[H],
        subdomains=[volume_subdomain, surface_subdomain],
        temperature=temperature_value,
    )
    my_model.t = fem.Constant(my_model.mesh.mesh, 0.0)
    dt = fem.Constant(test_mesh.mesh, 1.0)

    my_model.boundary_conditions = [
        F.ParticleFluxBC(subdomain=surface_subdomain, value=bc_value, species=H)
    ]

    my_model.define_temperature()
    my_model.define_function_spaces()
    my_model.assign_functions_to_species()
    my_model.define_meshtags_and_measures()
    my_model.define_boundary_conditions()

    for i in range(3):
        # RUN
        my_model.t.value += dt.value
        my_model.update_time_dependent_values()

        # TEST
        if isinstance(my_model.boundary_conditions[0].value_fenics, fem.Constant):
            computed_value = float(my_model.boundary_conditions[0].value_fenics)
            assert np.isclose(computed_value, expected_values[i])


def test_create_flux_values_fenics_multispecies():
    """Test that the create_flux_values_fenics method correctly sets the value_fenics
    attribute in a multispecies case"""
    # BUILD
    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    surface = F.SurfaceSubdomain1D(id=2, x=0)
    H, D = F.Species("H"), F.Species("D")
    my_model = F.HydrogenTransportProblem(
        mesh=test_mesh,
        temperature=10,
        subdomains=[my_vol, surface],
        species=[H, D],
    )
    my_model.t = fem.Constant(my_model.mesh.mesh, 4.0)

    my_bc_1 = F.ParticleFluxBC(subdomain=surface, value=lambda t: t + 1, species=H)
    my_bc_2 = F.ParticleFluxBC(subdomain=surface, value=lambda t: 2 * t + 3, species=D)
    my_model.boundary_conditions = [my_bc_1, my_bc_2]

    my_model.define_function_spaces()
    my_model.define_meshtags_and_measures()
    my_model.assign_functions_to_species()
    my_model.define_temperature()

    # RUN
    my_model.create_flux_values_fenics()

    # TEST
    assert np.isclose(float(my_model.boundary_conditions[0].value_fenics), 5)
    assert np.isclose(float(my_model.boundary_conditions[1].value_fenics), 11)
