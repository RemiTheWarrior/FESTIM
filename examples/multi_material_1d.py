import numpy as np

import festim as F

my_model = F.HydrogenTransportProblemDiscontinuous()
my_model.method_interface = "nietsche"

interface_1 = 0.5
interface_2 = 0.7

# for some reason if the mesh isn't fine enough then I have a random SEGV error
N = 2000
vertices = np.concatenate(
    [
        np.linspace(0, interface_1, num=N),
        np.linspace(interface_1, interface_2, num=N),
        np.linspace(interface_2, 1, num=N),
    ]
)

my_model.mesh = F.Mesh1D(vertices)

material_left = F.Material(D_0=2.0, E_D=0)
material_mid = F.Material(D_0=2.0, E_D=0)
material_right = F.Material(D_0=2.0, E_D=0)

material_left.K_S_0 = 2.0
material_left.E_K_S = 0
material_mid.K_S_0 = 4.0
material_mid.E_K_S = 0
material_right.K_S_0 = 6.0
material_right.E_K_S = 0

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

my_model.settings = F.Settings(atol=None, rtol=1e-5, transient=False)

my_model.exports = [
    F.VTXSpeciesExport(filename=f"u_{subdomain.id}.bp", field=H, subdomain=subdomain)
    for subdomain in my_model.volume_subdomains
]

my_model.initialise()
my_model.run()
