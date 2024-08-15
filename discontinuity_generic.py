from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py import PETSc
import basix
from festim.helpers_discontinuity import NewtonSolver, transfer_meshtags_to_submesh
import festim as F


# ---------------- Generate a mesh ----------------
def generate_mesh():
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)

    def top_boundary(x):
        return np.isclose(x[1], 1.0)

    def half(x):
        return x[1] <= 0.5 + 1e-14

    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 20, 20, dolfinx.mesh.CellType.triangle
    )

    # Split domain in half and set an interface tag of 5
    gdim = mesh.geometry.dim
    tdim = mesh.topology.dim
    fdim = tdim - 1
    top_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top_boundary)
    bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom_boundary)
    num_facets_local = (
        mesh.topology.index_map(fdim).size_local
        + mesh.topology.index_map(fdim).num_ghosts
    )
    facets = np.arange(num_facets_local, dtype=np.int32)
    values = np.full_like(facets, 0, dtype=np.int32)
    values[top_facets] = 1
    values[bottom_facets] = 2

    bottom_cells = dolfinx.mesh.locate_entities(mesh, tdim, half)
    num_cells_local = (
        mesh.topology.index_map(tdim).size_local
        + mesh.topology.index_map(tdim).num_ghosts
    )
    cells = np.full(num_cells_local, 4, dtype=np.int32)
    cells[bottom_cells] = 3
    ct = dolfinx.mesh.meshtags(
        mesh, tdim, np.arange(num_cells_local, dtype=np.int32), cells
    )
    all_b_facets = dolfinx.mesh.compute_incident_entities(
        mesh.topology, ct.find(3), tdim, fdim
    )
    all_t_facets = dolfinx.mesh.compute_incident_entities(
        mesh.topology, ct.find(4), tdim, fdim
    )
    interface = np.intersect1d(all_b_facets, all_t_facets)
    values[interface] = 5

    mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, facets, values)
    return mesh, mt, ct


mesh, mt, ct = generate_mesh()

top_domain = F.VolumeSubdomain(4, material=None)
bottom_domain = F.VolumeSubdomain(3, material=None)
list_of_subdomains = [bottom_domain, top_domain]
list_of_interfaces = {5: [bottom_domain, top_domain]}

gdim = mesh.geometry.dim
tdim = mesh.topology.dim
fdim = tdim - 1

num_facets_local = (
    mesh.topology.index_map(fdim).size_local + mesh.topology.index_map(fdim).num_ghosts
)

for subdomain in list_of_subdomains:
    subdomain.submesh, subdomain.submesh_to_mesh, subdomain.v_map = (
        dolfinx.mesh.create_submesh(mesh, tdim, ct.find(subdomain.id))[0:3]
    )

    subdomain.parent_to_submesh = np.full(num_facets_local, -1, dtype=np.int32)
    subdomain.parent_to_submesh[subdomain.submesh_to_mesh] = np.arange(
        len(subdomain.submesh_to_mesh), dtype=np.int32
    )

    # We need to modify the cell maps, as for `dS` integrals of interfaces between submeshes, there is no entity to map to.
    # We use the entity on the same side to fix this (as all restrictions are one-sided)

    # Transfer meshtags to submesh
    subdomain.ft, subdomain.facet_to_parent = transfer_meshtags_to_submesh(
        mesh, mt, subdomain.submesh, subdomain.v_map, subdomain.submesh_to_mesh
    )


# Hack, as we use one-sided restrictions, pad dS integral with the same entity from the same cell on both sides
# TODO ask Jorgen what this is for
mesh.topology.create_connectivity(fdim, tdim)
f_to_c = mesh.topology.connectivity(fdim, tdim)
for interface in list_of_interfaces:
    for facet in mt.find(interface):
        cells = f_to_c.links(facet)
        assert len(cells) == 2
        for domain in list_of_interfaces[interface]:
            map = domain.parent_to_submesh[cells]
            domain.parent_to_submesh[cells] = max(map)

# ._cpp_object needed on dolfinx 0.8.0
entity_maps = {
    subdomain.submesh: subdomain.parent_to_submesh
    for subdomain in list_of_subdomains
}


def D(T):
    k_B = 8.6173303e-5
    return 2 * ufl.exp(-0.1 / k_B / T)


def define_interior_eq(mesh, degree, submesh, submesh_to_mesh, value):
    # Compute map from parent entity to submesh cell
    codim = mesh.topology.dim - submesh.topology.dim
    ptdim = mesh.topology.dim - codim
    num_entities = (
        mesh.topology.index_map(ptdim).size_local
        + mesh.topology.index_map(ptdim).num_ghosts
    )
    mesh_to_submesh = np.full(num_entities, -1)
    mesh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh), dtype=np.int32)

    degree = 1
    element_CG = basix.ufl.element(
        basix.ElementFamily.P,
        submesh.basix_cell(),
        degree,
        basix.LagrangeVariant.equispaced,
    )
    element = basix.ufl.mixed_element([element_CG, element_CG])
    V = dolfinx.fem.functionspace(submesh, element)
    u = dolfinx.fem.Function(V)
    us = list(ufl.split(u))
    vs = list(ufl.TestFunctions(V))
    ct_r = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim,
        submesh_to_mesh,
        np.full_like(submesh_to_mesh, 1, dtype=np.int32),
    )
    val = dolfinx.fem.Constant(submesh, value)
    dx_r = ufl.Measure("dx", domain=mesh, subdomain_data=ct_r, subdomain_id=1)
    F = ufl.inner(ufl.grad(us[0]), ufl.grad(vs[0])) * dx_r - val * vs[0] * dx_r
    k = 2
    p = 0.1
    n = 0.5
    F += k * us[0] * (n - us[1]) * vs[1] * dx_r - p * us[1] * vs[1] * dx_r
    return u, vs, F, mesh_to_submesh


# for each subdomain, define the interior equation
for subdomain in list_of_subdomains:
    degree = 1
    subdomain.u, subdomain.vs, subdomain.F, subdomain.m_to_s = define_interior_eq(
        mesh, degree, subdomain.submesh, subdomain.submesh_to_mesh, 0.0
    )
    subdomain.u.name = f"u_{subdomain.id}"

# Add coupling term to the interface
# Get interface markers on submesh b
for interface in list_of_interfaces:
    subdomain_1 = list_of_interfaces[interface][0]
    subdomain_2 = list_of_interfaces[interface][1]

    dInterface = ufl.Measure(
        "dS", domain=mesh, subdomain_data=mt, subdomain_id=interface
    )
    b_res = "+"
    t_res = "-"

    v_b = subdomain_1.vs[0](b_res)
    v_t = subdomain_2.vs[0](t_res)

    u_bs = list(ufl.split(subdomain_1.u))
    u_ts = list(ufl.split(subdomain_2.u))
    u_b = u_bs[0](b_res)
    u_t = u_ts[0](t_res)

    def mixed_term(u, v, n):
        return ufl.dot(ufl.grad(u), n) * v

    n = ufl.FacetNormal(mesh)
    n_b = n(b_res)
    n_t = n(t_res)
    cr = ufl.Circumradius(mesh)
    h_b = 2 * cr(b_res)
    h_t = 2 * cr(t_res)
    gamma = 10.0

    # fabricate K
    W_0 = dolfinx.fem.functionspace(subdomain_1.submesh, ("DG", 0))
    K_0 = dolfinx.fem.Function(W_0)
    K_0.x.array[:] = 2
    W_1 = dolfinx.fem.functionspace(subdomain_2.submesh, ("DG", 0))
    K_1 = dolfinx.fem.Function(W_1)
    K_1.x.array[:] = 4

    K_b = K_0(b_res)
    K_t = K_1(t_res)

    F_0 = (
        -0.5 * mixed_term((u_b + u_t), v_b, n_b) * dInterface
        - 0.5 * mixed_term(v_b, (u_b / K_b - u_t / K_t), n_b) * dInterface
    )

    F_1 = (
        +0.5 * mixed_term((u_b + u_t), v_t, n_b) * dInterface
        - 0.5 * mixed_term(v_t, (u_b / K_b - u_t / K_t), n_b) * dInterface
    )
    F_0 += 2 * gamma / (h_b + h_t) * (u_b / K_b - u_t / K_t) * v_b * dInterface
    F_1 += -2 * gamma / (h_b + h_t) * (u_b / K_b - u_t / K_t) * v_t * dInterface

    subdomain_1.F += F_0
    subdomain_2.F += F_1

J = []
forms = []
for subdomain1 in list_of_subdomains:
    jac = []
    form = subdomain1.F
    for subdomain2 in list_of_subdomains:
        jac.append(
            dolfinx.fem.form(
                ufl.derivative(form, subdomain2.u), entity_maps=entity_maps
            )
        )
    J.append(jac)
    forms.append(dolfinx.fem.form(subdomain1.F, entity_maps=entity_maps))

# boundary conditions
b_bc = dolfinx.fem.Function(bottom_domain.u.function_space)
b_bc.x.array[:] = 0.2
bottom_domain.submesh.topology.create_connectivity(
    bottom_domain.submesh.topology.dim - 1, bottom_domain.submesh.topology.dim
)
bc_b = dolfinx.fem.dirichletbc(
    b_bc,
    dolfinx.fem.locate_dofs_topological(
        bottom_domain.u.function_space.sub(0), fdim, bottom_domain.ft.find(2)
    ),
)


t_bc = dolfinx.fem.Function(top_domain.u.function_space)
t_bc.x.array[:] = 0.05
top_domain.submesh.topology.create_connectivity(
    top_domain.submesh.topology.dim - 1, top_domain.submesh.topology.dim
)
bc_t = dolfinx.fem.dirichletbc(
    t_bc,
    dolfinx.fem.locate_dofs_topological(
        top_domain.u.function_space.sub(0), fdim, top_domain.ft.find(1)
    ),
)
bcs = [bc_b, bc_t]


solver = NewtonSolver(
    forms,
    J,
    [subdomain.u for subdomain in list_of_subdomains],
    bcs=bcs,
    max_iterations=10,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)
solver.solve(1e-5)

for subdomain in list_of_subdomains:
    u_sub_0 = subdomain.u.sub(0).collapse()
    u_sub_0.name = "u_sub_0"

    u_sub_1 = subdomain.u.sub(1).collapse()
    u_sub_1.name = "u_sub_1"
    bp = dolfinx.io.VTXWriter(
        mesh.comm, f"u_{subdomain.id}.bp", [u_sub_0, u_sub_1], engine="BP4"
    )
    bp.write(0)
    bp.close()


# derived quantities
V = dolfinx.fem.functionspace(mesh, ("CG", 1))
T = dolfinx.fem.Function(V)
T.interpolate(lambda x: 200 + x[1])


T_b = dolfinx.fem.Function(top_domain.u.sub(0).collapse().function_space)
T_b.interpolate(T)

ds_b = ufl.Measure("ds", domain=top_domain.submesh)
dx_b = ufl.Measure("dx", domain=bottom_domain.submesh)
dx = ufl.Measure("dx", domain=mesh)

n_b = ufl.FacetNormal(bottom_domain.submesh)

form = dolfinx.fem.form(bottom_domain.u.sub(0) * dx_b, entity_maps=entity_maps)
print(dolfinx.fem.assemble_scalar(form))

form = dolfinx.fem.form(T_b * ufl.dot(ufl.grad(bottom_domain.u.sub(0)), n_b) * ds_b, entity_maps=entity_maps)
print(dolfinx.fem.assemble_scalar(form))
