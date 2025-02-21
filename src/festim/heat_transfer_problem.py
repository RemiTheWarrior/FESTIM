import basix
import ufl
from dolfinx import fem
from dolfinx.io import VTXWriter

from festim import boundary_conditions, exports, helpers, problem
from festim import source as _source


class HeatTransferProblem(problem.ProblemBase):
    def __init__(
        self,
        mesh=None,
        subdomains=None,
        initial_condition=None,
        boundary_conditions=None,
        sources=None,
        exports=None,
        settings=None,
    ) -> None:
        super().__init__(
            mesh=mesh,
            sources=sources,
            exports=exports,
            subdomains=subdomains,
            boundary_conditions=boundary_conditions,
            settings=settings,
        )

        self.initial_condition = initial_condition
        self._vtxfile: VTXWriter | None = None

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, value):
        if not all(isinstance(source, _source.HeatSource) for source in value):
            raise TypeError("sources must be a list of festim.HeatSource objects")
        self._sources = value

    @property
    def boundary_conditions(self):
        return self._boundary_conditions

    @boundary_conditions.setter
    def boundary_conditions(self, value):
        if not all(
            isinstance(
                bc,
                (
                    boundary_conditions.FixedTemperatureBC,
                    boundary_conditions.HeatFluxBC,
                ),
            )
            for bc in value
        ):
            raise TypeError(
                "boundary_conditions must be a list of festim.FixedTemperatureBC or festim.HeatFluxBC objects"
            )
        self._boundary_conditions = value

    def initialise(self):
        self.define_function_space()
        self.define_meshtags_and_measures()

        self.t = fem.Constant(self.mesh.mesh, 0.0)
        if self.settings.transient:
            # TODO should raise error if no stepsize is provided
            # TODO Should this be an attribute of festim.Stepsize?
            self.dt = helpers.as_fenics_constant(
                self.settings.stepsize.initial_value, self.mesh.mesh
            )

        self.define_boundary_conditions()
        self.create_source_values_fenics()
        self.create_flux_values_fenics()
        self.create_initial_conditions()
        self.create_formulation()
        self.create_solver()
        self.initialise_exports()

    def define_function_space(self):
        """Creates the function space of the model, creates a mixed element if
        model is multispecies. Creates the main solution and previous solution
        function u and u_n. Create global DG function spaces of degree 0 and 1
        for the global diffusion coefficient"""

        degree = 1
        element_CG = basix.ufl.element(
            basix.ElementFamily.P,
            self.mesh.mesh.basix_cell(),
            degree,
            basix.LagrangeVariant.equispaced,
        )

        self.function_space = fem.functionspace(self.mesh.mesh, element_CG)

        self.u = fem.Function(self.function_space)
        self.u_n = fem.Function(self.function_space)
        self.test_function = ufl.TestFunction(self.function_space)

    def create_dirichletbc_form(self, bc):
        """Creates a dirichlet boundary condition form

        Args:
            bc (festim.FixedTemperatureBC): the boundary condition

        Returns:
            dolfinx.fem.bcs.DirichletBC: A representation of
                the boundary condition for modifying linear systems.
        """
        bc.create_value(
            function_space=self.function_space,
            t=self.t,
        )

        bc_dofs = bc.define_surface_subdomain_dofs(
            facet_meshtags=self.facet_meshtags,
            function_space=self.function_space,
        )

        if isinstance(bc.value_fenics, (fem.Function)):
            form = fem.dirichletbc(value=bc.value_fenics, dofs=bc_dofs)
        else:
            form = fem.dirichletbc(
                value=bc.value_fenics, dofs=bc_dofs, V=self.function_space
            )

        return form

    def create_source_values_fenics(self):
        """For each source create the value_fenics"""
        for source in self.sources:
            # create value_fenics for all source objects
            source.value.convert_input_value(
                mesh=self.mesh.mesh,
                function_space=self.function_space,
                t=self.t,
                up_to_ufl_expr=True,
            )

    def create_flux_values_fenics(self):
        """For each heat flux create the value_fenics"""
        for bc in self.boundary_conditions:
            # create value_fenics for all F.HeatFluxBC objects
            if isinstance(bc, boundary_conditions.HeatFluxBC):
                bc.create_value_fenics(
                    mesh=self.mesh.mesh,
                    temperature=self.u,
                    t=self.t,
                )

    def create_initial_conditions(self):
        """For each initial condition, create the value_fenics and assign it to
        the previous solution of the condition's species"""

        if not self.initial_condition:
            return

        if not self.settings.transient:
            raise ValueError(
                "Initial conditions can only be defined for transient simulations"
            )

        # create value_fenics for condition

        self.initial_condition.create_expr_fenics(
            mesh=self.mesh.mesh,
            function_space=self.function_space,
        )

        # assign to previous solution of species
        self.u_n.interpolate(self.initial_condition.expr_fenics)

    def create_formulation(self):
        """Creates the formulation of the model"""

        self.formulation = 0

        # add diffusion and time derivative for each species
        for vol in self.volume_subdomains:
            thermal_cond = vol.material.thermal_conductivity
            if callable(thermal_cond):
                thermal_cond = thermal_cond(self.u)

            self.formulation += ufl.dot(
                thermal_cond * ufl.grad(self.u), ufl.grad(self.test_function)
            ) * self.dx(vol.id)

            if self.settings.transient:
                density = vol.material.density
                heat_capacity = vol.material.heat_capacity
                if callable(density):
                    density = density(self.u)
                if callable(heat_capacity):
                    heat_capacity = heat_capacity(self.u)
                self.formulation += (
                    density
                    * heat_capacity
                    * ((self.u - self.u_n) / self.dt)
                    * self.test_function
                    * self.dx(vol.id)
                )

        # add sources
        for source in self.sources:
            self.formulation -= (
                source.value.fenics_object
                * self.test_function
                * self.dx(source.volume.id)
            )

        # add fluxes
        for bc in self.boundary_conditions:
            if isinstance(bc, boundary_conditions.HeatFluxBC):
                self.formulation -= (
                    bc.value_fenics * self.test_function * self.ds(bc.subdomain.id)
                )

    def initialise_exports(self):
        """Defines the export writers of the model, if field is given as
        a string, find species object in self.species"""

        for export in self.exports:
            if isinstance(export, exports.XDMFExport):
                raise NotImplementedError(
                    "XDMF export is not implemented yet for heat transfer problems"
                )
            if isinstance(export, exports.VTXTemperatureExport):
                self._vtxfile = VTXWriter(
                    self.u.function_space.mesh.comm,
                    export.filename,
                    [self.u],
                    engine="BP5",
                )

    def post_processing(self):
        """Post processes the model"""

        for export in self.exports:
            # TODO if export type derived quantity
            if isinstance(export, exports.SurfaceQuantity):
                raise NotImplementedError(
                    "SurfaceQuantity export is not implemented yet for heat transfer problems"
                )
                export.compute(
                    self.mesh.n,
                    self.ds,
                )
                # update export data
                export.t.append(float(self.t))

                # if filename given write export data to file
                if export.filename is not None:
                    export.write(t=float(self.t))
            if isinstance(export, exports.XDMFExport):
                export.write(float(self.t))

        if self._vtxfile is not None:
            self._vtxfile.write(float(self.t))

    def __del__(self):
        if self._vtxfile is not None:
            self._vtxfile.close()
