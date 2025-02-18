import numpy as np

import festim as F

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
my_model.surface_to_volume = {right_surface: subdomain, left_surface: subdomain}

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
    F.DirichletBC(left_surface, value=0.05, species=H),
    F.DirichletBC(right_surface, value=0.2, species=H),
]


my_model.temperature = lambda x: 300 + 100 * x[0]

my_model.settings = F.Settings(atol=None, rtol=1e-5, transient=False)

my_model.exports = [
    F.VTXSpeciesExport(filename=f"u_{subdomain.id}.bp", field=H, subdomain=subdomain)
    for subdomain in my_model.volume_subdomains
]

my_model.initialise()
my_model.run()
