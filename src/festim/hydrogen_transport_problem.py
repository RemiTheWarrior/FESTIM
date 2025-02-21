from collections.abc import Callable

from mpi4py import MPI

import basix
import dolfinx
import numpy.typing as npt
import tqdm.autonotebook
import ufl
from dolfinx import fem
from scifem import BlockedNewtonSolver

import festim.boundary_conditions
import festim.problem
from festim import (
    boundary_conditions,
    exports,
    k_B,
    problem,
)
from festim import (
    reaction as _reaction,
)
from festim import source as _source
from festim import (
    species as _species,
)
from festim import (
    subdomain as _subdomain,
)
from festim.helpers import as_fenics_constant
from festim.mesh import Mesh

__all__ = ["HTransportProblemDiscontinuous", "HydrogenTransportProblem"]


class HydrogenTransportProblem(problem.ProblemBase):
    """
    Hydrogen Transport Problem.

    Args:
        mesh: The mesh
        subdomains: List containing the subdomains
        species: List containing the species
        reactions: List containing the reactions
        temperature: The temperature or a function describing the temperature as
            a model of either space or space and time. Unit (K)
        sources: The hydrogen sources
        initial_conditions: The initial conditions
        boundary_conditions: The boundary conditions
        exports (list of festim.Export): the exports of the model
        traps (list of F.Trap): the traps of the model

    Attributes:
        mesh : The mesh
        subdomains: The subdomains
        species: The species
        reactions: the reaction
        temperature: The temperature in unit `K`
        sources: The hydrogen sources
        initial_conditions: The initial conditions
        boundary_conditions: List of Dirichlet boundary conditions
        exports (list of festim.Export): the export
        traps (list of F.Trap): the traps of the model
        dx (dolfinx.fem.dx): the volume measure of the model
        ds (dolfinx.fem.ds): the surface measure of the model
        function_space (dolfinx.fem.FunctionSpaceBase): the function space of the
            model
        facet_meshtags (dolfinx.mesh.MeshTags): the facet meshtags of the model
        volume_meshtags (dolfinx.mesh.MeshTags): the volume meshtags of the
            model
        formulation (ufl.form.Form): the formulation of the model
        solver (dolfinx.nls.newton.NewtonSolver): the solver of the model
        multispecies (bool): True if the model has more than one species.
        temperature_fenics (fem.Constant or fem.Function): the
            temperature of the model as a fenics object (fem.Constant or
            fem.Function).
        temperature_expr (fem.Expression): the expression of the temperature
            that is used to update the temperature_fenics
        temperature_time_dependent (bool): True if the temperature is time
            dependent
        V_DG_0 (dolfinx.fem.FunctionSpaceBase): A DG function space of degree 0
            over domain
        V_DG_1 (dolfinx.fem.FunctionSpaceBase): A DG function space of degree 1
            over domain
        volume_subdomains (list of festim.VolumeSubdomain): the volume subdomains
            of the model
        surface_subdomains (list of festim.SurfaceSubdomain): the surface subdomains
            of the model


    Examples:
        Can be used as either

        .. highlight:: python
        .. code-block:: python

            import festim as F
            my_model = F.HydrogenTransportProblem()
            my_model.mesh = F.Mesh(...)
            my_model.subdomains = [F.Subdomain(...)]
            my_model.species = [F.Species(name="H"), F.Species(name="Trap")]
            my_model.temperature = 500
            my_model.sources = [F.ParticleSource(...)]
            my_model.boundary_conditions = [F.BoundaryCondition(...)]
            my_model.initialise()

        or

        .. highlight:: python
        .. code-block:: python

            my_model = F.HydrogenTransportProblem(
                mesh=F.Mesh(...),
                subdomains=[F.Subdomain(...)],
                species=[F.Species(name="H"), F.Species(name="Trap")],
            )
            my_model.initialise()

    """

    def __init__(
        self,
        mesh: Mesh | None = None,
        subdomains: (
            list[_subdomain.VolumeSubdomain | _subdomain.SurfaceSubdomain] | None
        ) = None,
        species: list[_species.Species] | None = None,
        reactions: list[_reaction.Reaction] | None = None,
        temperature: (
            float
            | int
            | fem.Constant
            | fem.Function
            | Callable[
                [npt.NDArray[dolfinx.default_scalar_type]],
                npt.NDArray[dolfinx.default_scalar_type],
            ]
            | Callable[
                [npt.NDArray[dolfinx.default_scalar_type], fem.Constant],
                npt.NDArray[dolfinx.default_scalar_type],
            ]
            | None
        ) = None,
        sources=None,
        initial_conditions=None,
        boundary_conditions=None,
        settings=None,
        exports=None,
        traps=None,
        petsc_options=None,
    ):
        super().__init__(
            mesh=mesh,
            sources=sources,
            exports=exports,
            subdomains=subdomains,
            boundary_conditions=boundary_conditions,
            settings=settings,
            petsc_options=petsc_options,
        )

        self.species = species or []
        self.temperature = temperature
        self.reactions = reactions or []
        self.initial_conditions = initial_conditions or []
        self.traps = traps or []
        self.temperature_fenics = None
        self._vtxfiles: list[dolfinx.io.VTXWriter] = []

        self._element_for_traps = "DG"
        self.petcs_options = petsc_options

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if value is None:
            self._temperature = value
        elif isinstance(value, (float, int, fem.Constant, fem.Function)):
            self._temperature = value
        elif callable(value):
            self._temperature = value
        else:
            raise TypeError(
                "Value must be a float, int, fem.Constant, fem.Function, or callable"
            )

    @property
    def temperature_fenics(self):
        return self._temperature_fenics

    @temperature_fenics.setter
    def temperature_fenics(self, value):
        if value is None:
            self._temperature_fenics = value
            return
        elif not isinstance(
            value,
            (fem.Constant, fem.Function),
        ):
            raise TypeError("Value must be a fem.Constant or fem.Function")
        self._temperature_fenics = value

    @property
    def temperature_time_dependent(self):
        if self.temperature is None:
            return False
        if isinstance(self.temperature, fem.Constant):
            return False
        if callable(self.temperature):
            arguments = self.temperature.__code__.co_varnames
            return "t" in arguments
        else:
            return False

    @property
    def multispecies(self):
        return len(self.species) > 1

    @property
    def species(self) -> list[_species.Species]:
        return self._species

    @species.setter
    def species(self, value):
        # check that all species are of type festim.Species
        for spe in value:
            if not isinstance(spe, _species.Species):
                raise TypeError(
                    f"elements of species must be of type festim.Species not "
                    f"{type(spe)}"
                )
        self._species = value

    @property
    def facet_meshtags(self):
        return self._facet_meshtags

    @facet_meshtags.setter
    def facet_meshtags(self, value):
        if value is None:
            self._facet_meshtags = value
        elif isinstance(value, dolfinx.mesh.MeshTags):
            self._facet_meshtags = value
        else:
            raise TypeError("value must be of type dolfinx.mesh.MeshTags")

    @property
    def volume_meshtags(self):
        return self._volume_meshtags

    @volume_meshtags.setter
    def volume_meshtags(self, value):
        if value is None:
            self._volume_meshtags = value
        elif isinstance(value, dolfinx.mesh.MeshTags):
            self._volume_meshtags = value
        else:
            raise TypeError("value must be of type dolfinx.mesh.MeshTags")

    def initialise(self):
        self.create_species_from_traps()
        self.define_function_spaces()
        self.define_meshtags_and_measures()
        self.assign_functions_to_species()

        self.t = fem.Constant(self.mesh.mesh, 0.0)
        if self.settings.transient:
            # TODO should raise error if no stepsize is provided
            # TODO Should this be an attribute of festim.Stepsize?
            self.dt = as_fenics_constant(
                self.settings.stepsize.initial_value, self.mesh.mesh
            )

        self.create_implicit_species_value_fenics()

        self.define_temperature()
        self.define_boundary_conditions()
        self.create_source_values_fenics()
        self.create_flux_values_fenics()
        self.create_initial_conditions()
        self.create_formulation()
        self.create_solver()
        self.initialise_exports()

    def create_implicit_species_value_fenics(self):
        """For each implicit species, create the value_fenics"""
        for reaction in self.reactions:
            for reactant in reaction.reactant:
                if isinstance(reactant, _species.ImplicitSpecies):
                    reactant.create_value_fenics(
                        mesh=self.mesh.mesh,
                        t=self.t,
                    )

    def create_species_from_traps(self):
        """Generate a species and reaction per trap defined in self.traps"""

        for trap in self.traps:
            trap.create_species_and_reaction()
            self.species.append(trap.trapped_concentration)
            self.reactions.append(trap.reaction)

    def define_temperature(self):
        """Sets the value of temperature_fenics_value. The type depends on
        self.temperature. If self.temperature is a function on t only, create
        a fem.Constant. Else, create an dolfinx.fem.Expression (stored in
        self.temperature_expr) to be updated, a dolfinx.fem.Function object
        is created from the Expression (stored in self.temperature_fenics_value).
        Raise a ValueError if temperature is None.
        """
        # check if temperature is None
        if self.temperature is None:
            raise ValueError("the temperature attribute needs to be defined")

        # if temperature is a float or int, create a fem.Constant
        elif isinstance(self.temperature, (float, int)):
            self.temperature_fenics = as_fenics_constant(
                self.temperature, self.mesh.mesh
            )
        # if temperature is a fem.Constant or function, pass it to temperature_fenics
        elif isinstance(self.temperature, (fem.Constant, fem.Function)):
            self.temperature_fenics = self.temperature

        # if temperature is callable, process accordingly
        elif callable(self.temperature):
            arguments = self.temperature.__code__.co_varnames
            if "t" in arguments and "x" not in arguments:
                if not isinstance(self.temperature(t=float(self.t)), (float, int)):
                    raise ValueError(
                        f"self.temperature should return a float or an int, not "
                        f"{type(self.temperature(t=float(self.t)))} "
                    )
                # only t is an argument
                self.temperature_fenics = as_fenics_constant(
                    mesh=self.mesh.mesh, value=self.temperature(t=float(self.t))
                )
            else:
                x = ufl.SpatialCoordinate(self.mesh.mesh)
                degree = 1
                element_temperature = basix.ufl.element(
                    basix.ElementFamily.P,
                    self.mesh.mesh.basix_cell(),
                    degree,
                    basix.LagrangeVariant.equispaced,
                )
                function_space_temperature = fem.functionspace(
                    self.mesh.mesh, element_temperature
                )
                self.temperature_fenics = fem.Function(function_space_temperature)
                kwargs = {}
                if "t" in arguments:
                    kwargs["t"] = self.t
                if "x" in arguments:
                    kwargs["x"] = x

                # store the expression of the temperature
                # to update the temperature_fenics later
                self.temperature_expr = fem.Expression(
                    self.temperature(**kwargs),
                    function_space_temperature.element.interpolation_points(),
                )
                self.temperature_fenics.interpolate(self.temperature_expr)

    def initialise_exports(self):
        """Defines the export writers of the model, if field is given as
        a string, find species object in self.species"""

        for export in self.exports:
            # if name of species is given then replace with species object
            if isinstance(export.field, list):
                for idx, field in enumerate(export.field):
                    if isinstance(field, str):
                        export.field[idx] = _species.find_species_from_name(
                            field, self.species
                        )
            elif isinstance(export.field, str):
                export.field = _species.find_species_from_name(
                    export.field, self.species
                )

            # Initialize XDMFFile for writer
            if isinstance(export, exports.XDMFExport):
                export.define_writer(MPI.COMM_WORLD)
            if isinstance(export, exports.VTXSpeciesExport):
                functions = export.get_functions()
                self._vtxfiles.append(
                    dolfinx.io.VTXWriter(
                        functions[0].function_space.mesh.comm,
                        export.filename,
                        functions,
                        engine="BP5",
                    )
                )
        # compute diffusivity function for surface fluxes

        spe_to_D_global = {}  # links species to global D function
        spe_to_D_global_expr = {}  # links species to D expression

        for export in self.exports:
            if isinstance(export, exports.SurfaceQuantity):
                if export.field in spe_to_D_global:
                    # if already computed then use the same D
                    D = spe_to_D_global[export.field]
                    D_expr = spe_to_D_global_expr[export.field]
                else:
                    # compute D and add it to the dict
                    D, D_expr = self.define_D_global(export.field)
                    spe_to_D_global[export.field] = D
                    spe_to_D_global_expr[export.field] = D_expr

                # add the global D to the export
                export.D = D
                export.D_expr = D_expr

            # reset the data and time for SurfaceQuantity and VolumeQuantity
            if isinstance(export, (exports.SurfaceQuantity, exports.VolumeQuantity)):
                export.t = []
                export.data = []

    def define_D_global(self, species):
        """Defines the global diffusion coefficient for a given species

        Args:
            species (F.Species): the species

        Returns:
            dolfinx.fem.Function, dolfinx.fem.Expression: the global diffusion
                coefficient and the expression of the global diffusion coefficient
                for a given species
        """
        assert isinstance(species, _species.Species)

        D_0 = fem.Function(self.V_DG_0)
        E_D = fem.Function(self.V_DG_0)
        for vol in self.volume_subdomains:
            cell_indices = vol.locate_subdomain_entities(self.mesh.mesh)

            # replace values of D_0 and E_D by values from the material
            D_0.x.array[cell_indices] = vol.material.get_D_0(species=species)
            E_D.x.array[cell_indices] = vol.material.get_E_D(species=species)

        # create global D function
        D = fem.Function(self.V_DG_1)

        expr = D_0 * ufl.exp(
            -E_D / as_fenics_constant(k_B, self.mesh.mesh) / self.temperature_fenics
        )
        D_expr = fem.Expression(expr, self.V_DG_1.element.interpolation_points())
        D.interpolate(D_expr)
        return D, D_expr

    def define_function_spaces(self):
        """Creates the function space of the model, creates a mixed element if
        model is multispecies. Creates the main solution and previous solution
        function u and u_n. Create global DG function spaces of degree 0 and 1
        for the global diffusion coefficient"""

        # TODO: expose degree as a property to the user (element_degree ?) in ProblemBase
        degree = 1
        element_CG = basix.ufl.element(
            basix.ElementFamily.P,
            self.mesh.mesh.basix_cell(),
            degree,
            basix.LagrangeVariant.equispaced,
        )
        element_DG = basix.ufl.element(
            "DG",
            self.mesh.mesh.basix_cell(),
            degree,
            basix.LagrangeVariant.equispaced,
        )

        if not self.multispecies:
            element = element_CG
        else:
            elements = []
            for spe in self.species:
                if isinstance(spe, _species.Species):
                    if spe.mobile:
                        elements.append(element_CG)
                    elif self._element_for_traps == "DG":
                        elements.append(element_DG)
                    else:
                        elements.append(element_CG)
            element = basix.ufl.mixed_element(elements)

        self.function_space = fem.functionspace(self.mesh.mesh, element)

        # create global DG function spaces of degree 0 and 1
        element_DG0 = basix.ufl.element(
            "DG",
            self.mesh.mesh.basix_cell(),
            0,
            basix.LagrangeVariant.equispaced,
        )
        element_DG1 = basix.ufl.element(
            "DG",
            self.mesh.mesh.basix_cell(),
            1,
            basix.LagrangeVariant.equispaced,
        )
        self.V_DG_0 = fem.functionspace(self.mesh.mesh, element_DG0)
        self.V_DG_1 = fem.functionspace(self.mesh.mesh, element_DG1)

        self.u = fem.Function(self.function_space)
        self.u_n = fem.Function(self.function_space)

    def assign_functions_to_species(self):
        """Creates the solution, prev solution, test function and
        post-processing solution for each species, if model is multispecies,
        created a collapsed function space for each species"""

        if not self.multispecies:
            sub_solutions = [self.u]
            sub_prev_solution = [self.u_n]
            sub_test_functions = [ufl.TestFunction(self.function_space)]
            self.species[0].sub_function_space = self.function_space
            self.species[0].post_processing_solution = self.u
        else:
            sub_solutions = list(ufl.split(self.u))
            sub_prev_solution = list(ufl.split(self.u_n))
            sub_test_functions = list(ufl.TestFunctions(self.function_space))

            for idx, spe in enumerate(self.species):
                spe.sub_function_space = self.function_space.sub(idx)
                spe.post_processing_solution = self.u.sub(idx)
                spe.collapsed_function_space, _ = self.function_space.sub(
                    idx
                ).collapse()

        for idx, spe in enumerate(self.species):
            spe.solution = sub_solutions[idx]
            spe.prev_solution = sub_prev_solution[idx]
            spe.test_function = sub_test_functions[idx]

    def define_boundary_conditions(self):
        # @jhdark this all_bcs could be a property
        # I just don't want to modify self.boundary_conditions

        # create all_bcs which includes all flux bcs from SurfaceReactionBC
        all_bcs = self.boundary_conditions.copy()
        for bc in self.boundary_conditions:
            if isinstance(bc, boundary_conditions.SurfaceReactionBC):
                all_bcs += bc.flux_bcs
                all_bcs.remove(bc)

        for bc in all_bcs:
            if isinstance(bc.species, str):
                # if name of species is given then replace with species object
                bc.species = _species.find_species_from_name(bc.species, self.species)
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                bc.create_value_fenics(
                    mesh=self.mesh.mesh,
                    temperature=self.temperature_fenics,
                    t=self.t,
                )

        super().define_boundary_conditions()

    def create_dirichletbc_form(self, bc):
        """Creates a dirichlet boundary condition form

        Args:
            bc (festim.DirichletBC): the boundary condition

        Returns:
            dolfinx.fem.bcs.DirichletBC: A representation of
                the boundary condition for modifying linear systems.
        """
        # create value_fenics
        if not self.multispecies:
            function_space_value = bc.species.sub_function_space
        else:
            function_space_value = bc.species.collapsed_function_space

        bc.create_value(
            temperature=self.temperature_fenics,
            function_space=function_space_value,
            t=self.t,
        )

        # get dofs
        if self.multispecies and isinstance(bc.value_fenics, (fem.Function)):
            function_space_dofs = (
                bc.species.sub_function_space,
                bc.species.collapsed_function_space,
            )
        else:
            function_space_dofs = bc.species.sub_function_space

        bc_dofs = bc.define_surface_subdomain_dofs(
            facet_meshtags=self.facet_meshtags,
            function_space=function_space_dofs,
        )

        # create form
        if not self.multispecies and isinstance(bc.value_fenics, (fem.Function)):
            # no need to pass the functionspace since value_fenics is already a Function
            function_space_form = None
        else:
            function_space_form = bc.species.sub_function_space

        form = fem.dirichletbc(
            value=bc.value_fenics,
            dofs=bc_dofs,
            V=function_space_form,
        )

        return form

    def create_source_values_fenics(self):
        """For each source create the value_fenics"""
        for source in self.sources:
            # create value_fenics for all F.ParticleSource objects
            if isinstance(source, _source.ParticleSource):
                source.value.convert_input_value(
                    mesh=self.mesh.mesh,
                    function_space=self.function_space,
                    t=self.t,
                    temperature=self.temperature_fenics,
                    up_to_ufl_expr=True,
                )

    def create_flux_values_fenics(self):
        """For each particle flux create the value_fenics"""
        for bc in self.boundary_conditions:
            # create value_fenics for all F.ParticleFluxBC objects
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                bc.create_value_fenics(
                    mesh=self.mesh.mesh,
                    temperature=self.temperature_fenics,
                    t=self.t,
                )

    def create_initial_conditions(self):
        """For each initial condition, create the value_fenics and assign it to
        the previous solution of the condition's species"""

        if len(self.initial_conditions) > 0 and not self.settings.transient:
            raise ValueError(
                "Initial conditions can only be defined for transient simulations"
            )

        function_space_value = None

        for condition in self.initial_conditions:
            # create value_fenics for condition
            function_space_value = None
            if callable(condition.value):
                # if bc.value is a callable then need to provide a functionspace
                if not self.multispecies:
                    function_space_value = condition.species.sub_function_space
                else:
                    function_space_value = condition.species.collapsed_function_space

            condition.create_expr_fenics(
                mesh=self.mesh.mesh,
                temperature=self.temperature_fenics,
                function_space=function_space_value,
            )

            # assign to previous solution of species
            if not self.multispecies:
                condition.species.prev_solution.interpolate(condition.expr_fenics)
            else:
                idx = self.species.index(condition.species)
                self.u_n.sub(idx).interpolate(condition.expr_fenics)

    def create_formulation(self):
        """Creates the formulation of the model"""

        self.formulation = 0

        # add diffusion and time derivative for each species
        for spe in self.species:
            u = spe.solution
            u_n = spe.prev_solution
            v = spe.test_function

            for vol in self.volume_subdomains:
                D = vol.material.get_diffusion_coefficient(
                    self.mesh.mesh, self.temperature_fenics, spe
                )
                if spe.mobile:
                    self.formulation += ufl.dot(D * ufl.grad(u), ufl.grad(v)) * self.dx(
                        vol.id
                    )

                if self.settings.transient:
                    self.formulation += ((u - u_n) / self.dt) * v * self.dx(vol.id)

        for reaction in self.reactions:
            for reactant in reaction.reactant:
                if isinstance(reactant, festim.species.Species):
                    self.formulation += (
                        reaction.reaction_term(self.temperature_fenics)
                        * reactant.test_function
                        * self.dx(reaction.volume.id)
                    )

            # product
            if isinstance(reaction.product, list):
                products = reaction.product
            else:
                products = [reaction.product]
            for product in products:
                self.formulation += (
                    -reaction.reaction_term(self.temperature_fenics)
                    * product.test_function
                    * self.dx(reaction.volume.id)
                )
        # add sources
        for source in self.sources:
            self.formulation -= (
                source.value.fenics_object
                * source.species.test_function
                * self.dx(source.volume.id)
            )

        # add fluxes
        for bc in self.boundary_conditions:
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                self.formulation -= (
                    bc.value_fenics
                    * bc.species.test_function
                    * self.ds(bc.subdomain.id)
                )
            if isinstance(bc, boundary_conditions.SurfaceReactionBC):
                for flux_bc in bc.flux_bcs:
                    self.formulation -= (
                        flux_bc.value_fenics
                        * flux_bc.species.test_function
                        * self.ds(flux_bc.subdomain.id)
                    )

        # check if each species is defined in all volumes
        if not self.settings.transient:
            for spe in self.species:
                # if species mobile, already defined in diffusion term
                if not spe.mobile:
                    not_defined_in_volume = self.volume_subdomains.copy()
                    for vol in self.volume_subdomains:
                        # check reactions
                        for reaction in self.reactions:
                            if vol == reaction.volume:
                                if vol in not_defined_in_volume:
                                    not_defined_in_volume.remove(vol)

                    # add c = 0 to formulation where needed
                    for vol in not_defined_in_volume:
                        self.formulation += (
                            spe.solution * spe.test_function * self.dx(vol.id)
                        )

    def update_time_dependent_values(self):
        super().update_time_dependent_values()

        t = float(self.t)

        for reaction in self.reactions:
            for reactant in reaction.reactant:
                if isinstance(reactant, _species.ImplicitSpecies):
                    reactant.update_density(t=t)

        if not self.temperature_time_dependent:
            return

        if isinstance(self.temperature_fenics, fem.Constant):
            self.temperature_fenics.value = self.temperature(t=t)
        elif isinstance(self.temperature_fenics, fem.Function):
            self.temperature_fenics.interpolate(self.temperature_expr)

        for bc in self.boundary_conditions:
            if isinstance(
                bc,
                (
                    boundary_conditions.FixedConcentrationBC,
                    boundary_conditions.ParticleFluxBC,
                ),
            ):
                if bc.temperature_dependent:
                    bc.update(t=t)

        for source in self.sources:
            if source.value.temperature_dependent:
                source.value.update(t=t)

    def post_processing(self):
        """Post processes the model"""

        if self.temperature_time_dependent:
            # update global D if temperature time dependent or internal
            # variables time dependent
            species_not_updated = self.species.copy()  # make a copy of the species
            for export in self.exports:
                if isinstance(export, exports.SurfaceFlux):
                    # if the D of the species has not been updated yet
                    if export.field in species_not_updated:
                        export.D.interpolate(export.D_expr)
                        species_not_updated.remove(export.field)

        for export in self.exports:
            # TODO if export type derived quantity
            if isinstance(export, exports.SurfaceQuantity):
                if isinstance(
                    export,
                    (exports.SurfaceFlux, exports.TotalSurface, exports.AverageSurface),
                ):
                    export.compute(
                        self.ds,
                    )
                else:
                    export.compute()
                # update export data
                export.t.append(float(self.t))

                # if filename given write export data to file
                if export.filename is not None:
                    export.write(t=float(self.t))
            elif isinstance(export, exports.VolumeQuantity):
                if isinstance(export, (exports.TotalVolume, exports.AverageVolume)):
                    export.compute(self.dx)
                else:
                    export.compute()
                # update export data
                export.t.append(float(self.t))

                # if filename given write export data to file
                if export.filename is not None:
                    export.write(t=float(self.t))
            if isinstance(export, exports.XDMFExport):
                export.write(float(self.t))

        # should we move this to problem.ProblemBase?
        for vtxfile in self._vtxfiles:
            vtxfile.write(float(self.t))


class HTransportProblemDiscontinuous(HydrogenTransportProblem):
    interfaces: list[_subdomain.Interface]
    petsc_options: dict
    surface_to_volume: dict

    def __init__(
        self,
        mesh=None,
        subdomains=None,
        species=None,
        reactions=None,
        temperature=None,
        sources=None,
        initial_conditions=None,
        boundary_conditions=None,
        settings=None,
        exports=None,
        traps=None,
        interfaces: list[_subdomain.Interface] | None = None,
        surface_to_volume: dict | None = None,
        petsc_options: dict | None = None,
    ):
        """Class for a multi-material hydrogen transport problem
        For other arguments see ``festim.HydrogenTransportProblem``.

        Args:
            interfaces (list, optional): list of interfaces (``festim.Interface``
                objects). Defaults to None.
            surface_to_volume (dict, optional): correspondance dictionary linking
                each ``festim.SurfaceSubdomain`` objects to a ``festim.VolumeSubdomain``
                object). Defaults to None.
            petsc_options (dict, optional): petsc options to be passed to the
                ``festim.NewtonSolver`` object. If None, the default options are:
                ```
                default_petsc_options = {
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                }
                ```
                Defaults to None.
        """
        super().__init__(
            mesh,
            subdomains,
            species,
            reactions,
            temperature,
            sources,
            initial_conditions,
            boundary_conditions,
            settings,
            exports,
            traps,
        )
        self.interfaces = interfaces or []
        self.surface_to_volume = surface_to_volume or {}
        default_petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        self.petsc_options = petsc_options or default_petsc_options
        self._vtxfiles: list[dolfinx.io.VTXWriter] = []

    def initialise(self):
        # check that all species have a list of F.VolumeSubdomain as this is
        # different from F.HydrogenTransportProblem
        for spe in self.species:
            if not isinstance(spe.subdomains, list):
                raise TypeError("subdomains attribute should be list")

        self.define_meshtags_and_measures()

        # create submeshes and transfer meshtags to subdomains
        for subdomain in self.volume_subdomains:
            subdomain.create_subdomain(self.mesh.mesh, self.volume_meshtags)
            subdomain.transfer_meshtag(self.mesh.mesh, self.facet_meshtags)

        for interface in self.interfaces:
            interface.mt = self.volume_meshtags
            interface.parent_mesh = self.mesh.mesh

        self.create_species_from_traps()

        self.t = fem.Constant(self.mesh.mesh, 0.0)
        if self.settings.transient:
            # TODO should raise error if no stepsize is provided
            # TODO Should this be an attribute of festim.Stepsize?
            self.dt = as_fenics_constant(
                self.settings.stepsize.initial_value, self.mesh.mesh
            )

        self.create_implicit_species_value_fenics()

        self.define_temperature()
        self.create_source_values_fenics()
        self.create_flux_values_fenics()
        self.create_initial_conditions()
        for subdomain in self.volume_subdomains:
            self.define_function_spaces(subdomain)
            self.create_subdomain_formulation(subdomain)
            subdomain.u.name = f"u_{subdomain.id}"

        self.define_boundary_conditions()
        self.create_formulation()
        self.create_solver()
        self.initialise_exports()

    def create_dirichletbc_form(self, bc: boundary_conditions.FixedConcentrationBC):
        """
        Creates the ``value_fenics`` attribute for a given
        ``festim.FixedConcentrationBC`` and returns the appropriate
        ``dolfinx.fem.DirichletBC`` object.

        Args:
            bc (festim.FixedConcentrationBC): the dirichlet BC

        Returns:
            dolfinx.fem.DirichletBC: the appropriate dolfinx representation
                generated from ``dolfinx.fem.dirichletbc()``
        """
        fdim = self.mesh.mesh.topology.dim - 1
        volume_subdomain = self.surface_to_volume[bc.subdomain]
        sub_V = bc.species.subdomain_to_function_space[volume_subdomain]
        collapsed_V, _ = sub_V.collapse()

        bc.create_value(
            temperature=self.temperature_fenics,
            function_space=collapsed_V,
            t=self.t,
        )

        volume_subdomain.submesh.topology.create_connectivity(
            volume_subdomain.submesh.topology.dim - 1,
            volume_subdomain.submesh.topology.dim,
        )

        # mapping between sub_function space and collapsed is only needed if
        # value_fenics is a function of the collapsed space
        if isinstance(bc.value_fenics, fem.Function):
            function_space_dofs = (sub_V, collapsed_V)
        else:
            function_space_dofs = sub_V

        bc_dofs = dolfinx.fem.locate_dofs_topological(
            function_space_dofs,
            fdim,
            volume_subdomain.ft.find(bc.subdomain.id),
        )
        form = dolfinx.fem.dirichletbc(bc.value_fenics, bc_dofs, sub_V)
        return form

    def create_initial_conditions(self):
        if self.initial_conditions:
            raise NotImplementedError(
                "initial conditions not yet implemented for discontinuous"
            )

    def define_function_spaces(self, subdomain: _subdomain.VolumeSubdomain):
        """
        Creates appropriate function space and functions for a given subdomain (submesh)
        based on the number of species existing in this subdomain. Then stores the functionspace,
        the current solution (``u``) and the previous solution (``u_n``) functions. It also populates the
        correspondance dicts attributes of the species (eg. ``species.subdomain_to_solution``,
        ``species.subdomain_to_test_function``, etc) for easy access to the right subfunctions,
        sub-testfunctions etc.

        Args:
            subdomain (F.VolumeSubdomain): a subdomain of the geometry
        """
        # get number of species defined in the subdomain
        all_species = [
            species for species in self.species if subdomain in species.subdomains
        ]

        # instead of using the set function we use a list to keep the order
        unique_species = []
        for species in all_species:
            if species not in unique_species:
                unique_species.append(species)
        nb_species = len(unique_species)

        degree = 1
        element_CG = basix.ufl.element(
            basix.ElementFamily.P,
            subdomain.submesh.basix_cell(),
            degree,
            basix.LagrangeVariant.equispaced,
        )
        element = basix.ufl.mixed_element([element_CG] * nb_species)
        V = dolfinx.fem.functionspace(subdomain.submesh, element)
        u = dolfinx.fem.Function(V)
        u_n = dolfinx.fem.Function(V)

        # store attributes in the subdomain object
        subdomain.u = u
        subdomain.u_n = u_n

        # split the functions and assign the subfunctions to the species
        us = list(ufl.split(u))
        u_ns = list(ufl.split(u_n))
        vs = list(ufl.TestFunctions(V))
        for i, species in enumerate(unique_species):
            species.subdomain_to_solution[subdomain] = us[i]
            species.subdomain_to_prev_solution[subdomain] = u_ns[i]
            species.subdomain_to_test_function[subdomain] = vs[i]
            species.subdomain_to_function_space[subdomain] = V.sub(i)
            species.subdomain_to_post_processing_solution[subdomain] = u.sub(
                i
            ).collapse()
            species.subdomain_to_collapsed_function_space[subdomain] = V.sub(
                i
            ).collapse()
            name = f"{species.name}_{subdomain.id}"
            species.subdomain_to_post_processing_solution[subdomain].name = name

    def create_source_values_fenics(self):
        """For each source create the value_fenics"""
        for source in self.sources:
            # create value_fenics for all F.ParticleSource objects
            if isinstance(source, _source.ParticleSource):
                V = dolfinx.fem.functionspace(self.mesh.mesh, ("Lagrange", 1))

                source.value.convert_input_value(
                    mesh=self.mesh.mesh,
                    function_space=V,
                    t=self.t,
                    temperature=self.temperature_fenics,
                    up_to_ufl_expr=True,
                )

    def create_subdomain_formulation(self, subdomain: _subdomain.VolumeSubdomain):
        """
        Creates the variational formulation for each subdomain and stores it in ``subdomain.F``

        Args:
            subdomain (F.VolumeSubdomain): a subdomain of the geometry
        """
        form = 0
        # add diffusion and time derivative for each species
        for spe in self.species:
            if subdomain not in spe.subdomains:
                continue
            u = spe.subdomain_to_solution[subdomain]
            u_n = spe.subdomain_to_prev_solution[subdomain]
            v = spe.subdomain_to_test_function[subdomain]

            D = subdomain.material.get_diffusion_coefficient(
                self.mesh.mesh, self.temperature_fenics, spe
            )
            if self.settings.transient:
                form += ((u - u_n) / self.dt) * v * self.dx(subdomain.id)

            if spe.mobile:
                form += ufl.inner(D * ufl.grad(u), ufl.grad(v)) * self.dx(subdomain.id)

        # add reaction terms
        for reaction in self.reactions:
            if reaction.volume != subdomain:
                continue
            for species in reaction.reactant + reaction.product:
                if isinstance(species, festim.species.Species):
                    # TODO remove
                    # temporarily overide the solution to the one of the subdomain
                    species.solution = species.subdomain_to_solution[subdomain]

            # reactant
            for reactant in reaction.reactant:
                if isinstance(reactant, festim.species.Species):
                    form += (
                        reaction.reaction_term(self.temperature_fenics)
                        * reactant.subdomain_to_test_function[subdomain]
                        * self.dx(subdomain.id)
                    )

            # product
            if isinstance(reaction.product, list):
                products = reaction.product
            else:
                products = [reaction.product]
            for product in products:
                form += (
                    -reaction.reaction_term(self.temperature_fenics)
                    * product.subdomain_to_test_function[subdomain]
                    * self.dx(subdomain.id)
                )

        # add fluxes
        for bc in self.boundary_conditions:
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                # check that the bc is applied on a surface
                # belonging to this subdomain
                if subdomain == self.surface_to_volume[bc.subdomain]:
                    v = bc.species.subdomain_to_test_function[subdomain]
                    form -= bc.value_fenics * v * self.ds(bc.subdomain.id)

        # add volumetric sources
        for source in self.sources:
            v = source.species.subdomain_to_test_function[subdomain]
            if source.volume == subdomain:
                form -= source.value.fenics_object * v * self.dx(subdomain.id)

        # store the form in the subdomain object
        subdomain.F = form

    def create_formulation(self):
        """
        Takes all the formulations for each subdomain and adds the interface conditions.

        Finally compute the jacobian matrix and store it in the ``J`` attribute,
        adds the ``entity_maps`` to the forms and store them in the ``forms`` attribute
        """
        mesh = self.mesh.mesh
        mt = self.facet_meshtags

        for interface in self.interfaces:
            interface.mesh = mesh
            interface.mt = mt

        integral_data = [
            interface.compute_mapped_interior_facet_data(mesh)
            for interface in self.interfaces
        ]
        [interface.pad_parent_maps() for interface in self.interfaces]
        dInterface = ufl.Measure("dS", domain=mesh, subdomain_data=integral_data)

        def mixed_term(u, v, n):
            return ufl.dot(ufl.grad(u), n) * v

        n = ufl.FacetNormal(mesh)
        cr = ufl.Circumradius(mesh)

        entity_maps = {
            sd.submesh: sd.parent_to_submesh for sd in self.volume_subdomains
        }
        for interface in self.interfaces:
            gamma = interface.penalty_term

            subdomain_0, subdomain_1 = interface.subdomains
            res = interface.restriction
            n_0 = n(res[0])
            h_0 = 2 * cr(res[0])
            h_1 = 2 * cr(res[1])

            all_mobile_species = [spe for spe in self.species if spe.mobile]
            if len(all_mobile_species) > 1:
                raise NotImplementedError("Multiple mobile species not implemented")
            H = all_mobile_species[0]
            v_b = H.subdomain_to_test_function[subdomain_0](res[0])
            v_t = H.subdomain_to_test_function[subdomain_1](res[1])

            u_b = H.subdomain_to_solution[subdomain_0](res[0])
            u_t = H.subdomain_to_solution[subdomain_1](res[1])

            K_b = subdomain_0.material.get_solubility_coefficient(
                self.mesh.mesh, self.temperature_fenics(res[0]), H
            )
            K_t = subdomain_1.material.get_solubility_coefficient(
                self.mesh.mesh, self.temperature_fenics(res[1]), H
            )

            F_0 = -0.5 * mixed_term((u_b + u_t), v_b, n_0) * dInterface(
                interface.id
            ) - 0.5 * mixed_term(v_b, (u_b / K_b - u_t / K_t), n_0) * dInterface(
                interface.id
            )

            F_1 = +0.5 * mixed_term((u_b + u_t), v_t, n_0) * dInterface(
                interface.id
            ) - 0.5 * mixed_term(v_t, (u_b / K_b - u_t / K_t), n_0) * dInterface(
                interface.id
            )
            F_0 += (
                2
                * gamma
                / (h_0 + h_1)
                * (u_b / K_b - u_t / K_t)
                * v_b
                * dInterface(interface.id)
            )
            F_1 += (
                -2
                * gamma
                / (h_0 + h_1)
                * (u_b / K_b - u_t / K_t)
                * v_t
                * dInterface(interface.id)
            )

            subdomain_0.F += F_0
            subdomain_1.F += F_1

        J = []
        # this is the symbolic differentiation of the Jacobian
        for subdomain1 in self.volume_subdomains:
            jac = []
            for subdomain2 in self.volume_subdomains:
                jac.append(
                    ufl.derivative(subdomain1.F, subdomain2.u),
                )
            J.append(jac)
        # compile jacobian (J) and residual (F)
        self.forms = dolfinx.fem.form(
            [subdomain.F for subdomain in self.volume_subdomains],
            entity_maps=entity_maps,
            jit_options={
                "cffi_extra_compile_args": ["-O3", "-march=native"],
                "cffi_libraries": ["m"],
            },
        )
        self.J = dolfinx.fem.form(
            J,
            entity_maps=entity_maps,
            jit_options={
                "cffi_extra_compile_args": ["-O3", "-march=native"],
                "cffi_libraries": ["m"],
            },
        )

    def create_solver(self):
        self.solver = BlockedNewtonSolver(
            self.forms,
            [subdomain.u for subdomain in self.volume_subdomains],
            J=self.J,
            bcs=self.bc_forms,
            petsc_options=self.petsc_options,
        )
        self.solver.max_iterations = self.settings.max_iterations
        self.solver.convergence_criterion = self.settings.convergence_criterion
        self.solver.atol = self.settings.atol
        self.solver.rtol = self.settings.rtol

    def create_flux_values_fenics(self):
        """For each particle flux create the ``value_fenics`` attribute"""
        for bc in self.boundary_conditions:
            if isinstance(bc, boundary_conditions.ParticleFluxBC):
                volume_subdomain = self.surface_to_volume[bc.subdomain]
                bc.create_value_fenics(
                    mesh=volume_subdomain.submesh,
                    temperature=self.temperature_fenics,
                    t=self.t,
                )

    def initialise_exports(self):
        for export in self.exports:
            if isinstance(export, exports.VTXSpeciesExport):
                functions = export.get_functions()
                self._vtxfiles.append(
                    dolfinx.io.VTXWriter(
                        functions[0].function_space.mesh.comm,
                        export.filename,
                        functions,
                        engine="BP5",
                    )
                )
            else:
                raise NotImplementedError(f"Export type {type(export)} not implemented")

    def post_processing(self):
        # update post-processing solutions (for each species in each subdomain)
        # with new solution
        for subdomain in self.volume_subdomains:
            for species in self.species:
                if subdomain not in species.subdomains:
                    continue
                collapsed_function = species.subdomain_to_post_processing_solution[
                    subdomain
                ]
                u = subdomain.u
                v0_to_V = species.subdomain_to_collapsed_function_space[subdomain][1]
                collapsed_function.x.array[:] = u.x.array[v0_to_V]

        for vtxfile in self._vtxfiles:
            vtxfile.write(float(self.t))

        for export in self.exports:
            if not isinstance(export, exports.VTXSpeciesExport):
                raise NotImplementedError(f"Export type {type(export)} not implemented")

    def iterate(self):
        """Iterates the model for a given time step"""
        if self.show_progress_bar:
            self.progress_bar.update(
                min(self.dt.value, abs(self.settings.final_time - self.t.value))
            )
        self.t.value += self.dt.value

        self.update_time_dependent_values()

        # Solve main problem
        self.solver.solve()

        # post processing
        self.post_processing()

        # update previous solution
        for subdomain in self.volume_subdomains:
            subdomain.u_n.x.array[:] = subdomain.u.x.array[:]

        # adapt stepsize
        if self.settings.stepsize.adaptive:
            raise NotImplementedError("Adaptive stepsize not implemented")

    def run(self):
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
        else:
            # Solve steady-state
            self.solver.solve()
            self.post_processing()

    def __del__(self):
        for vtxfile in self._vtxfiles:
            vtxfile.close()


class HTransportProblemPenalty(HTransportProblemDiscontinuous):
    def create_formulation(self):
        """
        Takes all the formulations for each subdomain and adds the interface conditions.

        Finally compute the jacobian matrix and store it in the ``J`` attribute,
        adds the ``entity_maps`` to the forms and store them in the ``forms`` attribute
        """
        mesh = self.mesh.mesh
        mt = self.facet_meshtags

        for interface in self.interfaces:
            interface.mesh = mesh
            interface.mt = mt

        integral_data = [
            interface.compute_mapped_interior_facet_data(mesh)
            for interface in self.interfaces
        ]
        [interface.pad_parent_maps() for interface in self.interfaces]
        dInterface = ufl.Measure("dS", domain=mesh, subdomain_data=integral_data)

        entity_maps = {
            sd.submesh: sd.parent_to_submesh for sd in self.volume_subdomains
        }
        for interface in self.interfaces:
            subdomain_0, subdomain_1 = interface.subdomains
            res = interface.restriction

            all_mobile_species = [spe for spe in self.species if spe.mobile]
            if len(all_mobile_species) > 1:
                raise NotImplementedError("Multiple mobile species not implemented")
            H = all_mobile_species[0]
            v_b = H.subdomain_to_test_function[subdomain_0](res[0])
            v_t = H.subdomain_to_test_function[subdomain_1](res[1])

            u_b = H.subdomain_to_solution[subdomain_0](res[0])
            u_t = H.subdomain_to_solution[subdomain_1](res[1])

            K_b = subdomain_0.material.get_solubility_coefficient(
                self.mesh.mesh, self.temperature_fenics(res[0]), H
            )
            K_t = subdomain_1.material.get_solubility_coefficient(
                self.mesh.mesh, self.temperature_fenics(res[1]), H
            )

            if (
                subdomain_0.material.solubility_law
                == subdomain_1.material.solubility_law
            ):
                left = u_b / K_b
                right = u_t / K_t
            else:
                if subdomain_0.material.solubility_law == "henry":
                    left = u_b / K_b
                elif subdomain_0.material.solubility_law == "sievert":
                    left = (u_b / K_b) ** 2
                else:
                    raise ValueError(
                        f"Unknown material law {subdomain_0.material.solubility_law}"
                    )

                if subdomain_1.material.solubility_law == "henry":
                    right = u_t / K_t
                elif subdomain_1.material.solubility_law == "sievert":
                    right = (u_t / K_t) ** 2
                else:
                    raise ValueError(
                        f"Unknown material law {subdomain_1.material.solubility_law}"
                    )

            equality = right - left

            F_0 = (
                interface.penalty_term
                * ufl.inner(equality, v_b)
                * dInterface(interface.id)
            )
            F_1 = (
                -interface.penalty_term
                * ufl.inner(equality, v_t)
                * dInterface(interface.id)
            )

            subdomain_0.F += F_0
            subdomain_1.F += F_1

        J = []
        # this is the symbolic differentiation of the Jacobian
        for subdomain1 in self.volume_subdomains:
            jac = []
            for subdomain2 in self.volume_subdomains:
                jac.append(
                    ufl.derivative(subdomain1.F, subdomain2.u),
                )
            J.append(jac)
        # compile jacobian (J) and residual (F)
        self.forms = dolfinx.fem.form(
            [subdomain.F for subdomain in self.volume_subdomains],
            entity_maps=entity_maps,
        )
        self.J = dolfinx.fem.form(J, entity_maps=entity_maps)
