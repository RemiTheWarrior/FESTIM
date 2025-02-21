from importlib import metadata

try:
    __version__ = metadata.version("FESTIM")
except Exception:
    __version__ = "unknown"


R = 8.314462618  # Gas constant J.mol-1.K-1
k_B = 8.6173303e-5  # Boltzmann constant eV.K-1


from .boundary_conditions.dirichlet_bc import (
    DirichletBC,
    DirichletBCBase,
    FixedConcentrationBC,
    FixedTemperatureBC,
)
from .boundary_conditions.flux_bc import FluxBCBase, HeatFluxBC, ParticleFluxBC
from .boundary_conditions.henrys_bc import HenrysBC
from .boundary_conditions.flux_bc import FluxBCBase, ParticleFluxBC, HeatFluxBC
from .boundary_conditions.surface_reaction import SurfaceReactionBC

from .boundary_conditions.sieverts_bc import SievertsBC
from .exports.average_surface import AverageSurface
from .exports.average_volume import AverageVolume
from .exports.maximum_surface import MaximumSurface
from .exports.maximum_volume import MaximumVolume
from .exports.minimum_surface import MinimumSurface
from .exports.minimum_volume import MinimumVolume
from .exports.surface_flux import SurfaceFlux
from .exports.surface_quantity import SurfaceQuantity
from .exports.total_surface import TotalSurface
from .exports.total_volume import TotalVolume
from .exports.volume_quantity import VolumeQuantity
from .exports.vtx import VTXSpeciesExport, VTXTemperatureExport
from .exports.xdmf import XDMFExport
from .heat_transfer_problem import HeatTransferProblem
from .helpers import (
    as_fenics_constant,
    as_mapped_function,
    as_fenics_interp_expr_and_function,
    Value,
)
from .hydrogen_transport_problem import (
    HTransportProblemDiscontinuous,
    HydrogenTransportProblem,
    HTransportProblemPenalty,
)
from .problem_change_of_var import HydrogenTransportProblemDiscontinuousChangeVar
from .initial_condition import InitialCondition, InitialTemperature
from .material import Material
from .mesh.mesh import Mesh
from .mesh.mesh_1d import Mesh1D
from .mesh.mesh_from_xdmf import MeshFromXDMF
from .problem import ProblemBase
from .reaction import Reaction
from .settings import Settings
from .source import HeatSource, ParticleSource, SourceBase
from .species import ImplicitSpecies, Species, find_species_from_name, SpeciesChangeVar
from .stepsize import Stepsize
from .subdomain.interface import Interface
from .subdomain.surface_subdomain import SurfaceSubdomain, find_surface_from_id
from .subdomain.surface_subdomain_1d import SurfaceSubdomain1D
from .subdomain.volume_subdomain import VolumeSubdomain, find_volume_from_id
from .subdomain.volume_subdomain_1d import VolumeSubdomain1D
from .trap import Trap
