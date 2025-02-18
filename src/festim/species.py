from festim.subdomain.volume_subdomain import (
    VolumeSubdomain as _VolumeSubdomain,
)
from festim.helpers import as_fenics_constant

from typing import List, Union
import ufl
from dolfinx import fem


class Species:
    """
    Hydrogen species class for H transport simulation.

    Args:
        name (str, optional): a name given to the species. Defaults to None.
        mobile (bool, optional): whether the species is mobile or not.
            Defaults to True.
        subdomain (F.VolumeSubdomain, optional): the volume subdomain where the
            species is. Defaults to None.

    Attributes:
        name (str): a name given to the species.
        mobile (bool): whether the species is mobile or not.
        solution (dolfinx.fem.Function): the solution for the current timestep
        prev_solution (dolfinx.fem.Function): the solution for the previous
            timestep
        test_function (ufl.Argument): the testfunction associated with this
            species
        sub_function_space (dolfinx.fem.function.FunctionSpaceBase): the
            subspace of the function space
        collapsed_function_space (dolfinx.fem.function.FunctionSpaceBase): the
            collapsed function space for a species in the function space. In
            case single species case, this is None.
        post_processing_solution (dolfinx.fem.Function): the solution for post
            processing
        concentration (dolfinx.fem.Function): the concentration of the species
        subdomains (F.VolumeSubdomain): the volume subdomains where the species is
        subdomain_to_solution (dict): a dictionary mapping subdomains to solutions
        subdomain_to_prev_solution (dict): a dictionary mapping subdomains to
            previous solutions
        subdomain_to_test_function (dict): a dictionary mapping subdomains to
            test functions
        subdomain_to_post_processing_solution (dict): a dictionary mapping
            subdomains to post processing solutions
        subdomain_to_collapsed_function_space (dict): a dictionary mapping
            subdomains to collapsed function spaces
        subdomain_to_function_space (dict): a dictionary mapping subdomains to
            function spaces

    Usage:
        >>> from festim import Species, HydrogenTransportProblem
        >>> species = Species(name="H")
        >>> species.name
        'H'
        >>> my_model = HydrogenTransportProblem()
        >>> my_model.species.append(species)

    """

    subdomains: list[_VolumeSubdomain] | _VolumeSubdomain
    subdomain_to_solution: dict
    subdomain_to_prev_solution: dict
    subdomain_to_test_function: dict
    subdomain_to_post_processing_solution: dict
    subdomain_to_collapsed_function_space: dict
    subdomain_to_function_space: dict
    change_var: bool = False

    def __init__(self, name: str = None, mobile=True, subdomains=None) -> None:
        self.name = name
        self.mobile = mobile
        self.solution = None
        self.prev_solution = None
        self.test_function = None
        self.sub_function_space = None
        self.post_processing_solution = None
        self.collapsed_function_space = None

        self.subdomains = subdomains
        self.subdomain_to_solution = {}
        self.subdomain_to_prev_solution = {}
        self.subdomain_to_test_function = {}
        self.subdomain_to_post_processing_solution = {}
        self.subdomain_to_collapsed_function_space = {}
        self.subdomain_to_function_space = {}

    def __repr__(self) -> str:
        return f"Species({self.name})"

    def __str__(self) -> str:
        return f"{self.name}"

    @property
    def concentration(self):
        if self.change_var:
            return self._concentration
        else:
            return self.solution

    @concentration.setter
    def concentration(self, value):
        self._concentration = value

    @property
    def legacy(self) -> bool:
        """
        Check if we are using FESTIM 1.0 implementation or FESTIM 2.0
        """
        if not self.subdomain_to_solution:
            return True
        else:
            return False


class ImplicitSpecies:
    """Implicit species class for H transport simulation.
    c = n - others

    Args:
        n (Union[float, callable]): the total concentration of the species
        others (list[Species]): the list of species from which the implicit
            species concentration is computed (c = n - others)
        name (str, optional): a name given to the species. Defaults to None.

    Attributes:
        name (str): a name given to the species.
        n (float): the total concentration of the species
        others (list[Species]): the list of species from which the implicit
            species concentration is computed (c = n - others)
        concentration (form): the concentration of the species
        value_fenics: the total concentration as a fenics object
    """

    def __init__(
        self,
        n: Union[float, callable],
        others: List[Species] = None,
        name: str = None,
    ) -> None:
        self.name = name
        self.n = n
        self.others = others

    def __repr__(self) -> str:
        return f"ImplicitSpecies({self.name}, {self.n}, {self.others})"

    def __str__(self) -> str:
        return f"{self.name}"

    @property
    def concentration(self):
        if len(self.others) > 0:
            for other in self.others:
                if other.solution is None:
                    raise ValueError(
                        f"Cannot compute concentration of {self.name} "
                        + f"because {other.name} has no solution."
                    )
        return self.value_fenics - sum([other.solution for other in self.others])

    def create_value_fenics(self, mesh, t: fem.Constant):
        """Creates the value of the density as a fenics object and sets it to
        self.value_fenics.
        If the value is a constant, it is converted to a fenics.Constant.
        If the value is a function of t, it is converted to a fenics.Constant.
        Otherwise, it is converted to a ufl Expression

        Args:
            mesh (dolfinx.mesh.Mesh) : the mesh
            t (dolfinx.fem.Constant): the time
        """
        x = ufl.SpatialCoordinate(mesh)

        if isinstance(self.n, (int, float)):
            self.value_fenics = as_fenics_constant(mesh=mesh, value=self.n)

        elif isinstance(self.n, (fem.Function, ufl.core.expr.Expr)):
            self.value_fenics = self.n

        elif callable(self.n):
            arguments = self.n.__code__.co_varnames

            if "t" in arguments and "x" not in arguments and "T" not in arguments:
                # only t is an argument
                if not isinstance(self.n(t=float(t)), (float, int)):
                    raise ValueError(
                        f"self.value should return a float or an int, not {type(self.n(t=float(t)))} "
                    )
                self.value_fenics = as_fenics_constant(
                    mesh=mesh, value=self.n(t=float(t))
                )
            else:
                kwargs = {}
                if "t" in arguments:
                    kwargs["t"] = t
                if "x" in arguments:
                    kwargs["x"] = x

                self.value_fenics = self.n(**kwargs)

    def update_density(self, t):
        """Updates the density value (only if the density is a function of time only)

        Args:
            t (float): the time
        """
        if isinstance(self.n, (fem.Function, ufl.core.expr.Expr)):
            return

        if callable(self.n):
            arguments = self.n.__code__.co_varnames
            if isinstance(self.value_fenics, fem.Constant) and "t" in arguments:
                self.value_fenics.value = self.n(t=t)


def find_species_from_name(name: str, species: list):
    """Returns the correct species object from a list of species
    based on a string

    Args:
        name (str): the name of the species
        species (list): the list of species

    Returns:
        species (festim.Species): the species object with the correct name

    Raises:
        ValueError: if the species name is not found in the list of species

    """
    for spe in species:
        if spe.name == name:
            return spe
    raise ValueError(f"Species {name} not found in list of species")
