from mpi4py import MPI

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl

import festim as F

from .tools import error_L2


def generate_mesh(n=20):
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)

    def top_boundary(x):
        return np.isclose(x[1], 1.0)

    def half(x):
        return x[1] <= 0.5 + 1e-14

    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.triangle
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


def test_2_materials_2d_mms():
    """
    MMS case for a 2D problem with 2 materials
    adapted from https://festim-vv-report.readthedocs.io/en/v1.0/verification/mms/discontinuity.html
    """
    K_S_top = 3.0
    K_S_bot = 6.0
    D_top = 2.0
    D_bot = 5.0
    c_exact_top_ufl = (
        lambda x: 1 + ufl.sin(ufl.pi * (2 * x[0] + 0.5)) + ufl.cos(2 * ufl.pi * x[1])
    )

    def c_exact_bot_ufl(x):
        return K_S_bot / K_S_top * c_exact_top_ufl(x)

    c_exact_top_np = (
        lambda x: 1 + np.sin(np.pi * (2 * x[0] + 0.5)) + np.cos(2 * np.pi * x[1])
    )

    def c_exact_bot_np(x):
        return K_S_bot / K_S_top * c_exact_top_np(x)

    mesh, mt, ct = generate_mesh(100)

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.method_interface = "nietsche"
    my_model.mesh = F.Mesh(mesh)
    my_model.volume_meshtags = ct
    my_model.facet_meshtags = mt

    material_bottom = F.Material(D_0=D_bot, E_D=0, K_S_0=K_S_bot, E_K_S=0)
    material_top = F.Material(D_0=D_top, E_D=0, K_S_0=K_S_top, E_K_S=0)

    top_domain = F.VolumeSubdomain(4, material=material_top)
    bottom_domain = F.VolumeSubdomain(3, material=material_bottom)

    top_surface = F.SurfaceSubdomain(id=1)
    bottom_surface = F.SurfaceSubdomain(id=2)
    my_model.subdomains = [
        bottom_domain,
        top_domain,
        top_surface,
        bottom_surface,
    ]

    my_model.interfaces = [F.Interface(5, (bottom_domain, top_domain))]
    my_model.surface_to_volume = {
        top_surface: top_domain,
        bottom_surface: bottom_domain,
    }

    H = F.Species("H", mobile=True)

    my_model.species = [H]

    for species in my_model.species:
        species.subdomains = [bottom_domain, top_domain]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(top_surface, value=c_exact_top_ufl, species=H),
        F.FixedConcentrationBC(bottom_surface, value=c_exact_bot_ufl, species=H),
    ]

    source_top_val = (
        lambda x: 8
        * ufl.pi**2
        * (ufl.cos(2 * ufl.pi * x[0]) + ufl.cos(2 * ufl.pi * x[1]))
    )
    source_bottom_val = (
        lambda x: 40
        * ufl.pi**2
        * (ufl.cos(2 * ufl.pi * x[0]) + ufl.cos(2 * ufl.pi * x[1]))
    )
    my_model.sources = [
        F.ParticleSource(volume=top_domain, species=H, value=source_top_val),
        F.ParticleSource(volume=bottom_domain, species=H, value=source_bottom_val),
    ]

    my_model.temperature = 500.0  # lambda x: 300 + 10 * x[1] + 100 * x[0]

    my_model.settings = F.Settings(atol=1e-5, rtol=1e-5, transient=False)
    my_model.exports = [
        F.VTXSpeciesExport(f"u_{subdomain.id}.bp", field=H, subdomain=subdomain)
        for subdomain in my_model.volume_subdomains
    ]

    my_model.initialise()
    my_model.run()

    c_top_computed = H.subdomain_to_post_processing_solution[top_domain]
    c_bot_computed = H.subdomain_to_post_processing_solution[bottom_domain]

    L2_error_mobile = error_L2(c_top_computed, c_exact_top_np)
    L2_error_trapped = error_L2(c_bot_computed, c_exact_bot_np)

    assert L2_error_mobile < 1e-3
    assert L2_error_trapped < 1e-3


# TODO make this a MMS case
def test_1_material_discontinuous_version():
    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.method_interface = "nietsche"

    N = 1500
    vertices = np.linspace(0, 1, num=N)
    my_model.mesh = F.Mesh1D(vertices)

    material = F.Material(D_0=2.0, E_D=0)

    material.K_S_0 = 2.0
    material.E_K_S = 0

    subdomain = F.VolumeSubdomain1D(
        1, borders=[vertices[0], vertices[-1]], material=material
    )

    left_surface = F.SurfaceSubdomain1D(id=1, x=vertices[0])
    right_surface = F.SurfaceSubdomain1D(id=2, x=vertices[-1])

    my_model.subdomains = [subdomain, left_surface, right_surface]
    my_model.surface_to_volume = {
        right_surface: subdomain,
        left_surface: subdomain,
    }

    H = F.Species("H", mobile=True)
    trapped_H = F.Species("H_trapped", mobile=False)
    empty_trap = F.ImplicitSpecies(n=0.5, others=[trapped_H])

    my_model.species = [H, trapped_H]

    for species in my_model.species:
        species.subdomains = my_model.volume_subdomains

    my_model.reactions = [
        F.Reaction(
            reactant=[H, empty_trap],
            product=[trapped_H],
            k_0=2,
            E_k=0,
            p_0=0.1,
            E_p=0,
            volume=domain,
        )
        for domain in my_model.volume_subdomains
    ]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(left_surface, value=0.05, species=H),
        F.FixedConcentrationBC(right_surface, value=0.2, species=H),
    ]

    my_model.temperature = lambda x: 300 + 100 * x[0]

    my_model.settings = F.Settings(atol=1e-5, rtol=1e-5, transient=False)

    my_model.exports = [
        F.VTXSpeciesExport(
            filename=f"u_{subdomain.id}.bp", field=H, subdomain=subdomain
        )
        for subdomain in my_model.volume_subdomains
    ]

    my_model.initialise()
    my_model.run()


def test_3_materials_transient():
    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.method_interface = "nietsche"

    interface_1 = 0.5
    interface_2 = 0.7

    # for some reason if the mesh isn't fine enough then I have a random SEGV error
    N = 1500
    vertices = np.concatenate(
        [
            np.linspace(0, interface_1, num=N),
            np.linspace(interface_1, interface_2, num=N),
            np.linspace(interface_2, 1, num=N),
        ]
    )

    my_model.mesh = F.Mesh1D(vertices)

    material_left = F.Material(D_0=2.0, E_D=0, K_S_0=2.0, E_K_S=0)
    material_mid = F.Material(D_0=2.0, E_D=0, K_S_0=4.0, E_K_S=0)
    material_right = F.Material(D_0=2.0, E_D=0, K_S_0=6.0, E_K_S=0)

    left_domain = F.VolumeSubdomain1D(
        3, borders=[vertices[0], interface_1], material=material_left
    )
    middle_domain = F.VolumeSubdomain1D(
        4, borders=[interface_1, interface_2], material=material_mid
    )
    right_domain = F.VolumeSubdomain1D(
        5, borders=[interface_2, vertices[-1]], material=material_right
    )

    left_surface = F.SurfaceSubdomain1D(id=1, x=vertices[0])
    right_surface = F.SurfaceSubdomain1D(id=2, x=vertices[-1])

    # the ids here are arbitrary in 1D, you can put anything as long as it's not the same as the surfaces
    # TODO remove mesh and meshtags from these arguments
    my_model.interfaces = [
        F.Interface(6, (left_domain, middle_domain)),
        F.Interface(7, (middle_domain, right_domain)),
    ]

    my_model.subdomains = [
        left_domain,
        middle_domain,
        right_domain,
        left_surface,
        right_surface,
    ]
    my_model.surface_to_volume = {
        right_surface: right_domain,
        left_surface: left_domain,
    }

    H = F.Species("H", mobile=True)
    trapped_H = F.Species("H_trapped", mobile=False)
    empty_trap = F.ImplicitSpecies(n=0.5, others=[trapped_H])

    my_model.species = [H, trapped_H]

    for species in my_model.species:
        species.subdomains = my_model.volume_subdomains

    my_model.reactions = [
        F.Reaction(
            reactant=[H, empty_trap],
            product=[trapped_H],
            k_0=2,
            E_k=0,
            p_0=0.1,
            E_p=0,
            volume=domain,
        )
        for domain in [left_domain, middle_domain, right_domain]
    ]

    my_model.boundary_conditions = [
        F.DirichletBC(left_surface, value=0.05, species=H),
        F.DirichletBC(right_surface, value=0.2, species=H),
    ]

    my_model.temperature = lambda x: 300 + 100 * x[0]

    my_model.settings = F.Settings(
        atol=1e-10, rtol=1e-10, transient=True, final_time=100
    )
    my_model.settings.stepsize = 1

    my_model.exports = [
        F.VTXSpeciesExport(
            filename=f"u_{subdomain.id}.bp", field=H, subdomain=subdomain
        )
        for subdomain in my_model.volume_subdomains
    ] + [
        F.VTXSpeciesExport(
            filename=f"u_t_{subdomain.id}.bp",
            field=trapped_H,
            subdomain=subdomain,
        )
        for subdomain in my_model.volume_subdomains
    ]

    my_model.initialise()
    my_model.run()


def test_2_mats_particle_flux_bc():
    mesh, mt, ct = generate_mesh()

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.method_interface = "nietsche"
    my_model.mesh = F.Mesh(mesh)
    my_model.volume_meshtags = ct
    my_model.facet_meshtags = mt

    material_bottom = F.Material(D_0=1, E_D=0.1)
    material_top = F.Material(D_0=1, E_D=0.1)

    material_bottom.K_S_0 = 2.0
    material_bottom.E_K_S = 0.1
    material_top.K_S_0 = 4.0
    material_top.E_K_S = 0.1

    top_domain = F.VolumeSubdomain(4, material=material_top)
    bottom_domain = F.VolumeSubdomain(3, material=material_bottom)

    top_surface = F.SurfaceSubdomain(id=1)
    bottom_surface = F.SurfaceSubdomain(id=2)
    my_model.subdomains = [
        bottom_domain,
        top_domain,
        top_surface,
        bottom_surface,
    ]

    # we should be able to automate this
    my_model.interfaces = [F.Interface(5, (bottom_domain, top_domain))]
    my_model.surface_to_volume = {
        top_surface: top_domain,
        bottom_surface: bottom_domain,
    }

    H = F.Species("H", mobile=True)

    my_model.species = [H]

    for species in my_model.species:
        species.subdomains = [bottom_domain, top_domain]

    my_model.boundary_conditions = [
        F.DirichletBC(top_surface, value=0.05, species=H),
        F.ParticleFluxBC(bottom_surface, value=lambda x: 1.0 + x[0], species=H),
    ]

    my_model.temperature = lambda x: 300 + 10 * x[1] + 100 * x[0]

    my_model.settings = F.Settings(atol=1e-5, rtol=1e-5, transient=False)

    my_model.exports = [
        F.VTXSpeciesExport(f"u_{subdomain.id}.bp", field=H, subdomain=subdomain)
        for subdomain in my_model.volume_subdomains
    ]

    my_model.initialise()
    my_model.run()
