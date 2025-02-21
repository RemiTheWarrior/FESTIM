import os

import mpi4py.MPI as MPI

import dolfinx
import numpy as np
import pytest
import tqdm.autonotebook
import ufl
from dolfinx import fem
from dolfinx.io import XDMFFile

import festim as F


def source_from_exact_solution(
    exact_solution, thermal_conductivity, density, heat_capacity
):
    import sympy as sp
    from sympy.vector import CoordSys3D, divergence, gradient

    R = CoordSys3D("R")
    x = [R.x, R.y, R.z]
    t = sp.symbols("t")

    u = exact_solution(x, t)
    density = density(x, t)
    heat_capacity = heat_capacity(x, t)
    thermal_cond = thermal_conductivity(x, t)
    source = density * heat_capacity * sp.diff(u, t) - divergence(
        thermal_cond * gradient(u, x), x
    )

    return sp.lambdify([x, t], source)


def error_L2(u_computed, u_exact, degree_raise=3):
    # Create higher order function space
    degree = u_computed.function_space.ufl_element().degree
    family = u_computed.function_space.ufl_element().family_name
    mesh = u_computed.function_space.mesh
    W = fem.functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = fem.Function(W)
    u_W.interpolate(u_computed)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = fem.Function(W)
    if isinstance(u_exact, ufl.core.expr.Expr):
        u_expr = fem.Expression(u_exact, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_exact)

    # Compute the error in the higher order function space
    e_W = fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def test_MMS_1():
    thermal_conductivity = 4.0

    def exact_solution(x):
        return 2 * x[0] ** 2

    mms_source = -4 * thermal_conductivity

    my_problem = F.HeatTransferProblem()

    my_problem.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 2000))
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)
    mat = F.Material(D_0=None, E_D=None)
    mat.thermal_conductivity = thermal_conductivity

    my_problem.subdomains = [
        left,
        right,
        F.VolumeSubdomain1D(id=1, borders=[0, 1], material=mat),
    ]

    my_problem.boundary_conditions = [
        F.FixedTemperatureBC(subdomain=left, value=exact_solution),
        F.FixedTemperatureBC(subdomain=right, value=exact_solution),
    ]

    my_problem.sources = [
        F.HeatSource(value=mms_source, volume=my_problem.volume_subdomains[0])
    ]

    my_problem.settings = F.Settings(
        atol=1e-10,
        rtol=1e-10,
        transient=False,
    )

    my_problem.initialise()
    my_problem.run()

    computed_solution = my_problem.u
    L2_error = error_L2(computed_solution, exact_solution)
    assert L2_error < 1e-7


def test_MMS_T_dependent_thermal_cond():
    """MMS test with space T dependent thermal cond"""

    def thermal_conductivity(T):
        return 3 * T + 2

    def exact_solution(x):
        return 2 * x[0] ** 2 + 1

    # TODO would be nice to automate
    def mms_source(x):
        return -(72 * x[0] ** 2 + 20)

    my_problem = F.HeatTransferProblem()

    my_problem.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 2000))
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)
    mat = F.Material(D_0=None, E_D=None)
    mat.thermal_conductivity = thermal_conductivity

    my_problem.subdomains = [
        left,
        right,
        F.VolumeSubdomain1D(id=1, borders=[0, 1], material=mat),
    ]
    my_problem.boundary_conditions = [
        F.FixedTemperatureBC(subdomain=left, value=exact_solution),
        F.FixedTemperatureBC(subdomain=right, value=exact_solution),
    ]

    my_problem.sources = [
        F.HeatSource(value=mms_source, volume=my_problem.volume_subdomains[0])
    ]

    my_problem.settings = F.Settings(
        atol=1e-10,
        rtol=1e-10,
        transient=False,
    )

    my_problem.initialise()
    my_problem.run()

    computed_solution = my_problem.u
    L2_error = error_L2(computed_solution, exact_solution)
    assert L2_error < 1e-7


def test_heat_transfer_transient():
    """
    MMS test for transient heat transfer
    constant thermal conductivity density and heat capacity
    """
    density = 2
    heat_capacity = 3
    thermal_conductivity = 4

    def exact_solution(x, t):
        return 2 * x[0] ** 2 + 20 * t

    dTdt = 20
    mms_source = density * heat_capacity * dTdt - thermal_conductivity * 4

    my_problem = F.HeatTransferProblem()

    my_problem.mesh = F.Mesh1D(vertices=np.linspace(2, 3, 2000))
    left = F.SurfaceSubdomain1D(id=1, x=2)
    right = F.SurfaceSubdomain1D(id=2, x=3)
    mat = F.Material(D_0=None, E_D=None)
    mat.thermal_conductivity = thermal_conductivity
    mat.density = density
    mat.heat_capacity = heat_capacity

    my_problem.subdomains = [
        left,
        right,
        F.VolumeSubdomain1D(id=1, borders=[2, 3], material=mat),
    ]
    # NOTE: it's good to check that without the IC the solution is not the exact one
    my_problem.initial_condition = F.InitialTemperature(lambda x: exact_solution(x, 0))

    my_problem.boundary_conditions = [
        F.FixedTemperatureBC(subdomain=left, value=exact_solution),
        F.FixedTemperatureBC(subdomain=right, value=exact_solution),
    ]

    my_problem.sources = [
        F.HeatSource(value=mms_source, volume=my_problem.volume_subdomains[0])
    ]

    my_problem.settings = F.Settings(
        atol=1e-8,
        rtol=1e-10,
        final_time=1,  # final time shouldn't be too long so that a potential error at the initial timestep is not negligible
    )

    # Forward euler isn't great so dt should be small
    # although it's ok here since the time derivative is constant
    my_problem.settings.stepsize = F.Stepsize(0.1)

    my_problem.exports = [
        F.VTXTemperatureExport(filename="test_transient_heat_transfer.bp")
    ]

    my_problem.initialise()
    my_problem.run()

    computed_solution = my_problem.u
    # we use the exact final time of the simulation which may differ from the one specified in the settings
    final_time_sim = my_problem.t.value

    def exact_solution_end(x):
        return exact_solution(x, final_time_sim)

    L2_error = error_L2(computed_solution, exact_solution_end)
    assert L2_error < 1e-7


def test_MES():
    """Method of Exact Solution test for transient heat transfer
    with thermal cond. k = 2, T = 0 on surfaces, and source term q = 8 k

    Analytical solution: T(x) = 4 x (1 - x)
    """
    my_problem = F.HeatTransferProblem()

    my_problem.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 3000))
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)
    mat = F.Material(D_0=None, E_D=None)
    mat.thermal_conductivity = 2

    volume_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=mat)
    my_problem.subdomains = [left, right, volume_subdomain]

    my_problem.boundary_conditions = [
        F.FixedTemperatureBC(subdomain=left, value=0),
        F.FixedTemperatureBC(subdomain=right, value=0),
    ]

    my_problem.sources = [
        F.HeatSource(value=8 * mat.thermal_conductivity, volume=volume_subdomain)
    ]

    my_problem.settings = F.Settings(
        atol=1e-10,
        rtol=1e-10,
        transient=False,
    )

    my_problem.initialise()
    my_problem.run()

    computed_solution = my_problem.u

    def analytical_solution(x):
        return 4 * x[0] * (1 - x[0])

    L2_error = error_L2(computed_solution, analytical_solution)
    assert L2_error < 1e-7


# TODO populate this in other tests
def test_sympify():
    def exact_solution(x, t):
        return 2 * x[0] ** 2 + 20 * t

    def density(T):
        return 0.2 * T + 2

    def heat_capacity(T):
        return 0.2 * T + 3

    def thermal_conductivity(T):
        return 0.1 * T + 4

    mms_source_from_sp = source_from_exact_solution(
        exact_solution,
        density=lambda x, t: density(exact_solution(x, t)),
        heat_capacity=lambda x, t: heat_capacity(exact_solution(x, t)),
        thermal_conductivity=lambda x, t: thermal_conductivity(exact_solution(x, t)),
    )

    def mms_source(x, t):
        return mms_source_from_sp((x[0], None, None), t)

    my_problem = F.HeatTransferProblem()

    my_problem.mesh = F.Mesh1D(vertices=np.linspace(2, 3, 2100))
    left = F.SurfaceSubdomain1D(id=1, x=2)
    right = F.SurfaceSubdomain1D(id=2, x=3)
    mat = F.Material(D_0=None, E_D=None)
    mat.thermal_conductivity = thermal_conductivity
    mat.density = density
    mat.heat_capacity = heat_capacity

    my_problem.subdomains = [
        left,
        right,
        F.VolumeSubdomain1D(id=1, borders=[2, 3], material=mat),
    ]

    # NOTE: it's good to check that without the IC the solution is not the exact one
    my_problem.initial_condition = F.InitialTemperature(lambda x: exact_solution(x, 0))

    my_problem.boundary_conditions = [
        F.FixedTemperatureBC(subdomain=left, value=exact_solution),
        F.FixedTemperatureBC(subdomain=right, value=exact_solution),
    ]

    my_problem.sources = [
        F.HeatSource(value=mms_source, volume=my_problem.volume_subdomains[0])
    ]

    my_problem.settings = F.Settings(
        atol=1e-8,
        rtol=1e-10,
        final_time=1,  # final time shouldn't be too long so that a potential error at the initial timestep is not negligible
    )

    # Forward euler isn't great so dt should be small
    # although it's ok here since the time derivative is constant
    my_problem.settings.stepsize = F.Stepsize(0.05)

    my_problem.exports = [
        F.VTXTemperatureExport(filename="test_transient_heat_transfer.bp")
    ]

    my_problem.initialise()
    my_problem.run()

    computed_solution = my_problem.u
    # we use the exact final time of the simulation which may differ from the one specified in the settings
    final_time_sim = my_problem.t.value

    def exact_solution_end(x):
        return exact_solution(x, final_time_sim)

    L2_error = error_L2(computed_solution, exact_solution_end)
    assert L2_error < 1e-7


def test_sources():
    """Tests the sources setter of the HeatTransferProblem class"""
    htp = F.HeatTransferProblem()
    vol = F.VolumeSubdomain1D(1, borders=[0, 1], material=None)
    # Test that setting valid sources works
    valid_sources = [
        F.HeatSource(value=1, volume=vol),
        F.HeatSource(value=1, volume=vol),
    ]
    htp.sources = valid_sources
    assert htp.sources == valid_sources

    # Test that setting invalid sources raises a TypeError
    with pytest.raises(TypeError, match="festim.HeatSource objects"):
        spe = F.Species("H")
        htp.sources = [
            F.ParticleSource(1, vol, spe),
            F.ParticleSource(1, vol, spe),
        ]


def test_boundary_conditions():
    """Tests the boundary_conditions setter of the HeatTransferProblem class"""
    htp = F.HeatTransferProblem()
    left = F.SurfaceSubdomain(1)

    # Test that setting valid boundary conditions works
    valid_bcs = [F.FixedTemperatureBC(left, 0), F.FixedTemperatureBC(left, 0)]
    htp.boundary_conditions = valid_bcs
    assert htp.boundary_conditions == valid_bcs

    # Test that setting invalid boundary conditions raises a TypeError
    with pytest.raises(TypeError):
        htp.boundary_conditions = [F.FixedConcentrationBC(left, 0, 0)]


mesh_1D = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
mesh_2D = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
mesh_3D = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)


@pytest.mark.parametrize("mesh", [mesh_1D, mesh_2D, mesh_3D])
def test_meshtags_from_xdmf(tmp_path, mesh):
    """Test that the facet and volume meshtags are read correctly from the mesh XDMF files"""
    # create mesh functions
    fdim = mesh.topology.dim - 1
    vdim = mesh.topology.dim

    # create facet meshtags
    facet_indices = []
    for i in range(vdim):
        # add the boundary entities at 0 and 1 in each dimension
        facets_zero = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[i], 0)
        )
        facets_one = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[i], 1)
        )

        facet_indices += [facets_zero, facets_one]

    facet_tags = []

    for idx, _ in enumerate(facet_indices):
        # add tags for each boundary
        facet_tag = np.full(len(facet_indices[i]), idx + 1, dtype=np.int32)
        facet_tags.append(facet_tag)

    facet_tags = np.array(facet_tags).flatten()
    facet_indices = np.array(facet_indices).flatten()

    facet_meshtags = dolfinx.mesh.meshtags(mesh, fdim, facet_indices, facet_tags)

    # create volume meshtags
    num_cells = mesh.topology.index_map(vdim).size_local
    mesh_cell_indices = np.arange(num_cells, dtype=np.int32)
    # tag all volumes with 0
    tags_volumes = np.full(num_cells, 0, dtype=np.int32)
    # create 2 volumes for x<0.5 and x>0.5
    volume_indices_left = dolfinx.mesh.locate_entities(
        mesh,
        vdim,
        lambda x: x[0] <= 0.5,
    )

    volume_indices_right = dolfinx.mesh.locate_entities(
        mesh,
        vdim,
        lambda x: x[0] >= 0.5,
    )
    tags_volumes[volume_indices_left] = 2
    tags_volumes[volume_indices_right] = 3

    volume_meshtags = dolfinx.mesh.meshtags(mesh, vdim, mesh_cell_indices, tags_volumes)

    # write files
    surface_file_path = os.path.join(tmp_path, "facets_file.xdmf")
    surface_file = XDMFFile(MPI.COMM_WORLD, surface_file_path, "w")
    surface_file.write_mesh(mesh)
    surface_file.write_meshtags(facet_meshtags, mesh.geometry)

    volume_file_path = os.path.join(tmp_path, "volumes_file.xdmf")
    volume_file = XDMFFile(MPI.COMM_WORLD, volume_file_path, "w")
    volume_file.write_mesh(mesh)
    volume_file.write_meshtags(volume_meshtags, mesh.geometry)

    # read files
    my_model = F.HeatTransferProblem(
        mesh=F.MeshFromXDMF(
            volume_file=volume_file_path,
            facet_file=surface_file_path,
            mesh_name="mesh",
            surface_meshtags_name="mesh_tags",
            volume_meshtags_name="mesh_tags",
        )
    )
    my_model.define_meshtags_and_measures()

    # TEST
    assert volume_meshtags.dim == my_model.volume_meshtags.dim
    assert volume_meshtags.values.all() == my_model.volume_meshtags.values.all()
    assert facet_meshtags.dim == my_model.facet_meshtags.dim
    assert facet_meshtags.values.all() == my_model.facet_meshtags.values.all()


def test_raise_error_non_unique_vol_ids():
    """Test that an error is raised if the volume ids are not unique"""
    my_problem = F.HeatTransferProblem()
    my_problem.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 2000))
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)
    mat = F.Material(D_0=None, E_D=None)
    mat.thermal_conductivity = 1

    my_problem.subdomains = [
        left,
        right,
        F.VolumeSubdomain1D(id=1, borders=[0, 0.5], material=mat),
        F.VolumeSubdomain1D(id=1, borders=[0.5, 1], material=mat),
    ]

    # read files
    with pytest.raises(ValueError):
        my_problem.define_meshtags_and_measures()


def test_initial_condition():
    """Test that an error is raised when initial conditions are defined for a transient simulation"""
    my_problem = F.HeatTransferProblem()

    my_problem.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 3000))
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)
    mat = F.Material(D_0=None, E_D=None)
    mat.thermal_conductivity = 2

    volume_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=mat)
    my_problem.subdomains = [left, right, volume_subdomain]

    my_problem.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_problem.initial_condition = F.InitialTemperature(100)

    with pytest.raises(
        ValueError,
        match="Initial conditions can only be defined for transient simulations",
    ):
        my_problem.initialise()


test_mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
dummy_mat = F.Material(D_0=1, E_D=1, name="dummy_mat")
dummy_mat.thermal_conductivity = 1
dummy_mat.density = 1
dummy_mat.heat_capacity = 1


def test_adaptive_timestepping_grows():
    """Tests that the stepsize grows"""
    # BUILD
    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    my_model = F.HeatTransferProblem(
        mesh=test_mesh,
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=True, final_time=10),
        subdomains=[my_vol],
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
    for i in range(10):
        my_model.iterate()

        # check that the current value is greater than the previous one
        assert my_model.dt.value > previous_value

        previous_value = float(my_model.dt)


def test_adaptive_timestepping_shrinks():
    """Tests that the stepsize shrinks"""
    # BUILD
    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    my_model = F.HeatTransferProblem(
        mesh=test_mesh,
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=True, final_time=10),
        subdomains=[my_vol],
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
    for i in range(10):
        my_model.iterate()

        # check that the current value is smaller than the previous one
        assert my_model.dt.value < previous_value

        previous_value = float(my_model.dt)


@pytest.mark.parametrize(
    "bc_value, expected_values",
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
def test_update_time_dependent_values_HeatFluxBC(bc_value, expected_values):
    """Test that time dependent fluxes are updated at each time step,
    and match an expected value"""
    # BUILD
    my_vol = F.VolumeSubdomain1D(id=1, borders=[0, 4], material=dummy_mat)
    surface = F.SurfaceSubdomain1D(id=2, x=0)

    my_model = F.HeatTransferProblem(mesh=test_mesh, subdomains=[my_vol, surface])
    my_model.t = fem.Constant(my_model.mesh.mesh, 0.0)
    dt = fem.Constant(test_mesh.mesh, 1.0)

    my_bc = F.HeatFluxBC(subdomain=surface, value=bc_value)
    my_model.boundary_conditions = [my_bc]

    my_model.define_function_space()

    # dummy contant value to test temperature depedence
    my_model.u = F.as_fenics_constant(value=10, mesh=test_mesh.mesh)

    my_model.define_meshtags_and_measures()
    my_model.create_initial_conditions()
    my_model.create_flux_values_fenics()

    for i in range(3):
        # RUN
        my_model.t.value += dt.value
        my_model.update_time_dependent_values()

        # TEST
        if isinstance(my_model.boundary_conditions[0].value_fenics, fem.Constant):
            computed_value = float(my_model.boundary_conditions[0].value_fenics)
            assert np.isclose(computed_value, expected_values[i])
