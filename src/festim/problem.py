from typing import Any

from mpi4py import MPI

import numpy as np
import tqdm.autonotebook
import ufl
from dolfinx import fem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc

import festim as F
from festim.mesh.mesh import Mesh as _Mesh
from festim.source import SourceBase as _SourceBase
from festim.subdomain.volume_subdomain import (
    VolumeSubdomain as _VolumeSubdomain,
)


class ProblemBase:
    """
    Base class for :py:class:`HeatTransferProblem <festim.heat_transfer_problem.HeatTransferProblem>` and
    :py:class:`HydrogenTransportProblem <festim.hydrogen_transport_problem.HydrogenTransportProblem>`.

    Attributes:
        show_progress_bar: If `True` a progress bar is displayed during the simulation
        progress_bar: the progress bar
    """

    mesh: _Mesh
    sources: list[_SourceBase]
    exports: list[Any]
    subdomains: list[_VolumeSubdomain]
    show_progress_bar: bool
    progress_bar: None | tqdm.autonotebook.tqdm

    def __init__(
        self,
        mesh: _Mesh = None,
        sources=None,
        exports=None,
        subdomains=None,
        boundary_conditions=None,
        settings=None,
        petsc_options=None,
    ) -> None:
        self.mesh = mesh
        # for arguments to initialise as empty list
        # if arg not None, assign arg, else assign empty list
        self.subdomains = subdomains or []
        self.boundary_conditions = boundary_conditions or []
        self.sources = sources or []
        self.exports = exports or []
        self.settings = settings

        self.dx = None
        self.ds = None
        self.function_space = None
        self.facet_meshtags = None
        self.volume_meshtags = None
        self.formulation = None
        self.bc_forms = []
        self.show_progress_bar = True
        self.petsc_options = petsc_options

    @property
    def volume_subdomains(self):
        return [s for s in self.subdomains if isinstance(s, F.VolumeSubdomain)]

    @property
    def surface_subdomains(self):
        return [s for s in self.subdomains if isinstance(s, F.SurfaceSubdomain)]

    def define_meshtags_and_measures(self):
        """Defines the facet and volume meshtags of the model which are used
        to define the measures fo the model, dx and ds"""

        if isinstance(self.mesh, F.MeshFromXDMF):
            # TODO: fix naming inconsistency between facet and surface meshtags
            self.facet_meshtags = self.mesh.define_surface_meshtags()
            self.volume_meshtags = self.mesh.define_volume_meshtags()

        elif (
            isinstance(self.mesh, F.Mesh)
            and self.facet_meshtags is None
            and self.volume_meshtags is None
        ):
            self.facet_meshtags, self.volume_meshtags = self.mesh.define_meshtags(
                surface_subdomains=self.surface_subdomains,
                volume_subdomains=self.volume_subdomains,
                # if self has attribute interfaces pass it
                interfaces=getattr(self, "interfaces", None),
            )

        # check volume ids are unique
        vol_ids = [vol.id for vol in self.volume_subdomains]
        if len(vol_ids) != len(np.unique(vol_ids)):
            raise ValueError("Volume ids are not unique")

        # define measures
        self.ds = ufl.Measure(
            "ds", domain=self.mesh.mesh, subdomain_data=self.facet_meshtags
        )
        self.dx = ufl.Measure(
            "dx", domain=self.mesh.mesh, subdomain_data=self.volume_meshtags
        )

    def define_boundary_conditions(self):
        """Defines the dirichlet boundary conditions of the model"""
        for bc in self.boundary_conditions:
            if isinstance(bc, F.DirichletBCBase):
                form = self.create_dirichletbc_form(bc)
                self.bc_forms.append(form)

    def create_solver(self):
        """Creates the solver of the model"""
        problem = fem.petsc.NonlinearProblem(
            self.formulation,
            self.u,
            bcs=self.bc_forms,
        )
        self.solver = NewtonSolver(MPI.COMM_WORLD, problem)
        self.solver.atol = self.settings.atol
        self.solver.rtol = self.settings.rtol
        self.solver.max_it = self.settings.max_iterations

        ksp = self.solver.krylov_solver

        if self.petsc_options is None:
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
            ksp.setErrorIfNotConverged(True)
        else:
            # Set PETSc options
            opts = PETSc.Options()
            option_prefix = ksp.getOptionsPrefix()
            for k, v in self.petsc_options.items():
                opts[f"{option_prefix}{k}"] = v
            ksp.setFromOptions()

    def run(self):
        """Runs the model"""

        if self.settings.transient:
            # Solve transient
            if self.show_progress_bar:
                self.progress_bar = tqdm.autonotebook.tqdm(
                    desc=f"Solving {self.__class__.__name__}",
                    total=self.settings.final_time,
                    unit_scale=True,
                )
            while self.t.value < self.settings.final_time:
                self.iterate()
            if self.show_progress_bar:
                self.progress_bar.refresh()  # refresh progress bar to show 100%
                self.progress_bar.close()
        else:
            # Solve steady-state
            self.solver.solve(self.u)
            self.post_processing()

    def iterate(self):
        """Iterates the model for a given time step"""
        if self.show_progress_bar:
            self.progress_bar.update(
                min(self.dt.value, abs(self.settings.final_time - self.t.value))
            )
        self.t.value += self.dt.value

        self.update_time_dependent_values()

        # solve main problem
        nb_its, converged = self.solver.solve(self.u)

        # post processing
        self.post_processing()

        # update previous solution
        self.u_n.x.array[:] = self.u.x.array[:]

        # adapt stepsize
        if self.settings.stepsize.adaptive:
            new_stepsize = self.settings.stepsize.modify_value(
                value=self.dt.value, nb_iterations=nb_its, t=self.t.value
            )
            self.dt.value = new_stepsize

    def update_time_dependent_values(self):
        t = float(self.t)
        for bc in self.boundary_conditions:
            if bc.time_dependent:
                bc.update(t=t)

        for source in self.sources:
            if source.value.time_dependent:
                source.value.update(t=t)
