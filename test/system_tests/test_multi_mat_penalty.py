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
        lambda x: 3
        + ufl.sin(ufl.pi * (2 * x[1] + 0.5))
        + 0.1 * ufl.cos(2 * ufl.pi * x[0])
    )
    c_exact_top_np = (
        lambda x: 3 + np.sin(np.pi * (2 * x[1] + 0.5)) + 0.1 * np.cos(2 * np.pi * x[0])
    )

    def c_exact_bot_ufl(x):
        return K_S_bot / K_S_top**2 * c_exact_top_ufl(x) ** 2

    def c_exact_bot_np(x):
        return K_S_bot / K_S_top**2 * c_exact_top_np(x) ** 2

    mesh, mt, ct = generate_mesh(100)

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.method_interface = "penalty"
    my_model.mesh = F.Mesh(mesh)
    my_model.volume_meshtags = ct
    my_model.facet_meshtags = mt

    material_top = F.Material(D_0=D_top, E_D=0, K_S_0=K_S_top, E_K_S=0)
    material_bottom = F.Material(D_0=D_bot, E_D=0, K_S_0=K_S_bot, E_K_S=0)

    material_top.solubility_law = "sievert"
    material_bottom.solubility_law = "henry"

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

    my_model.interfaces = [
        F.Interface(5, (bottom_domain, top_domain), penalty_term=1),
    ]
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

    x = ufl.SpatialCoordinate(mesh)

    source_top_val = -ufl.div(D_top * ufl.grad(c_exact_top_ufl(x)))
    source_bottom_val = -ufl.div(D_bot * ufl.grad(c_exact_bot_ufl(x)))
    my_model.sources = [
        F.ParticleSource(volume=top_domain, species=H, value=source_top_val),
        F.ParticleSource(volume=bottom_domain, species=H, value=source_bottom_val),
    ]

    my_model.temperature = 500.0  # lambda x: 300 + 10 * x[1] + 100 * x[0]

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)
    my_model.exports = [
        F.VTXSpeciesExport(f"u_{subdomain.id}.bp", field=H, subdomain=subdomain)
        for subdomain in my_model.volume_subdomains
    ]

    my_model.initialise()
    my_model.run()

    c_top_computed = H.subdomain_to_post_processing_solution[top_domain]
    c_bot_computed = H.subdomain_to_post_processing_solution[bottom_domain]

    L2_error_top = error_L2(c_top_computed, c_exact_top_np)
    L2_error_bot = error_L2(c_bot_computed, c_exact_bot_np)

    assert L2_error_top < 1e-3
    assert L2_error_bot < 1e-3
