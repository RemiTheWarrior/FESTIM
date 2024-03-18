import festim as F
import numpy as np
from dolfinx import fem
import ufl
from .tools import error_L2
from dolfinx.mesh import meshtags, create_unit_square, create_unit_cube, locate_entities
from mpi4py import MPI

test_mesh_1d = F.Mesh1D(np.linspace(0, 1, 10000))
test_mesh_2d = create_unit_square(MPI.COMM_WORLD, 50, 50)
test_mesh_3d = create_unit_cube(MPI.COMM_WORLD, 20, 20, 20)
x_1d = ufl.SpatialCoordinate(test_mesh_1d.mesh)
x_2d = ufl.SpatialCoordinate(test_mesh_2d)
x_3d = ufl.SpatialCoordinate(test_mesh_3d)


def test_1_mobile_MMS_steady_state():
    """
    MMS test with one mobile species at steady state
    """

    def u_exact(mod):
        return lambda x: 1 + mod.sin(2 * mod.pi * x[0])

    H_analytical_ufl = u_exact(ufl)
    H_analytical_np = u_exact(np)

    elements = ufl.FiniteElement("CG", test_mesh_1d.mesh.ufl_cell(), 1)
    V = fem.FunctionSpace(test_mesh_1d.mesh, elements)
    T = fem.Function(V)

    D_0 = 1
    E_D = 0.1
    T_expr = lambda x: 500 + 100 * x[0]
    T.interpolate(T_expr)
    D = D_0 * ufl.exp(-E_D / (F.k_B * T))

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d
    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)

    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=H_analytical_ufl, species=H),
        F.DirichletBC(subdomain=right, value=H_analytical_ufl, species=H),
    ]

    f = -ufl.div(D * ufl.grad(H_analytical_ufl(x_1d)))
    my_model.sources = [F.Source(value=f, volume=vol, species=H)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, H_analytical_np)

    assert L2_error < 1e-7


def test_1_mobile_MMS_transient():
    """
    MMS test with 1 mobile species in 0.1s transient, the value at the last time step is
    compared to an analytical solution
    """

    final_time = 0.1

    def u_exact(mod):
        return lambda x, t: 1 + mod.sin(2 * mod.pi * x[0]) + 2 * t**2

    def u_exact_alt(mod):
        return lambda x: u_exact(mod)(x, final_time)

    H_analytical_ufl = u_exact(ufl)
    H_analytical_np = u_exact_alt(np)

    elements = ufl.FiniteElement("P", test_mesh_1d.mesh.ufl_cell(), 1)
    V = fem.FunctionSpace(test_mesh_1d.mesh, elements)
    T = fem.Function(V)

    D_0 = 1
    E_D = 0.1
    T_expr = lambda x: 600 + 50 * x[0]
    T.interpolate(T_expr)
    D = D_0 * ufl.exp(-E_D / (F.k_B * T))

    # FESTIM model

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d
    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)

    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=H_analytical_ufl, species=H),
        F.DirichletBC(subdomain=right, value=H_analytical_ufl, species=H),
    ]

    init_value = lambda x: 1 + ufl.sin(2 * ufl.pi * x[0])
    my_model.initial_conditions = [F.InitialCondition(value=init_value, species=H)]

    f = lambda x, t: 4 * t - ufl.div(D * ufl.grad(H_analytical_ufl(x, t)))
    my_model.sources = [F.Source(value=f, volume=vol, species=H)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=final_time)
    my_model.settings.stepsize = final_time / 50

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, H_analytical_np)

    assert L2_error < 5e-4


def test_1_mobile_MMS_2D():
    """Tests that a steady simulation can be run in a 2D domain with
    1 mobile species"""

    def u_exact(mod):
        return lambda x: 1 + mod.sin(2 * mod.pi * x[0]) + mod.cos(2 * mod.pi * x[1])

    H_analytical_ufl = u_exact(ufl)
    H_analytical_np = u_exact(np)

    elements = ufl.FiniteElement("CG", test_mesh_2d.ufl_cell(), 1)
    V = fem.FunctionSpace(test_mesh_2d, elements)
    T = fem.Function(V)

    D_0 = 1
    E_D = 0.1
    T_expr = lambda x: 500 + 100 * x[0]
    T.interpolate(T_expr)
    D = D_0 * ufl.exp(-E_D / (F.k_B * T))

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh(mesh=test_mesh_2d)

    # create facet meshtags
    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[0], 1)),
    ]
    facet_indices, facet_markers = [], []
    fdim = test_mesh_2d.topology.dim - 1
    for marker, locator in boundaries:
        facets = locate_entities(test_mesh_2d, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    my_facet_meshtags = meshtags(
        test_mesh_2d, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )

    # create volume meshtags
    vdim = test_mesh_2d.topology.dim
    num_cells = test_mesh_2d.topology.index_map(vdim).size_local
    mesh_cell_indices = np.arange(num_cells, dtype=np.int32)
    tags_volumes = np.full(num_cells, 1, dtype=np.int32)
    my_volume_meshtags = meshtags(test_mesh_2d, vdim, mesh_cell_indices, tags_volumes)

    my_model.facet_meshtags = my_facet_meshtags
    my_model.volume_meshtags = my_volume_meshtags

    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain(id=1, material=my_mat)
    left = F.SurfaceSubdomain(id=1)
    right = F.SurfaceSubdomain(id=2)

    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=H_analytical_ufl, species=H),
        F.DirichletBC(subdomain=right, value=H_analytical_ufl, species=H),
    ]

    f = -ufl.div(D * ufl.grad(H_analytical_ufl(x_2d)))
    my_model.sources = [F.Source(value=f, volume=vol, species=H)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, H_analytical_np)

    assert L2_error < 2e-3


def test_1_mobile_MMS_3D():
    """Tests that a steady simulation can be run in a 3D domain with
    1 mobile species"""

    def u_exact(mod):
        return lambda x: 1 + mod.sin(2 * mod.pi * x[0]) + mod.cos(2 * mod.pi * x[1])

    H_analytical_ufl = u_exact(ufl)
    H_analytical_np = u_exact(np)

    elements = ufl.FiniteElement("CG", test_mesh_3d.ufl_cell(), 1)
    V = fem.FunctionSpace(test_mesh_3d, elements)
    T = fem.Function(V)

    D_0 = 1
    E_D = 0.1
    T_expr = lambda x: 500 + 100 * x[0]
    T.interpolate(T_expr)
    D = D_0 * ufl.exp(-E_D / (F.k_B * T))

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh(mesh=test_mesh_3d)

    # create facet meshtags
    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[0], 1)),
    ]
    facet_indices, facet_markers = [], []
    fdim = test_mesh_3d.topology.dim - 1
    for marker, locator in boundaries:
        facets = locate_entities(test_mesh_3d, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    my_facet_meshtags = meshtags(
        test_mesh_3d, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )

    # create volume meshtags
    vdim = test_mesh_3d.topology.dim
    num_cells = test_mesh_3d.topology.index_map(vdim).size_local
    mesh_cell_indices = np.arange(num_cells, dtype=np.int32)
    tags_volumes = np.full(num_cells, 1, dtype=np.int32)
    my_volume_meshtags = meshtags(test_mesh_3d, vdim, mesh_cell_indices, tags_volumes)

    my_model.facet_meshtags = my_facet_meshtags
    my_model.volume_meshtags = my_volume_meshtags

    my_mat = F.Material(name="mat", D_0=D_0, E_D=E_D)
    vol = F.VolumeSubdomain(id=1, material=my_mat)
    left = F.SurfaceSubdomain(id=1)
    right = F.SurfaceSubdomain(id=2)

    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    my_model.species = [H]

    my_model.temperature = T_expr

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=H_analytical_ufl, species=H),
        F.DirichletBC(subdomain=right, value=H_analytical_ufl, species=H),
    ]

    f = -ufl.div(D * ufl.grad(H_analytical_ufl(x_3d)))
    my_model.sources = [F.Source(value=f, volume=vol, species=H)]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

    H_computed = H.post_processing_solution

    L2_error = error_L2(H_computed, H_analytical_np)

    assert L2_error < 1e-2
