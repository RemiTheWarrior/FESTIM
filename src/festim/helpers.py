from collections.abc import Callable
from typing import Optional

import dolfinx
import numpy as np
import ufl
from dolfinx import fem


def as_fenics_constant(
    value: float | int | fem.Constant, mesh: dolfinx.mesh.Mesh
) -> fem.Constant:
    """Converts a value to a dolfinx.Constant

    Args:
        value: the value to convert
        mesh: the mesh of the domiain

    Returns:
        The converted value

    Raises:
        TypeError: if the value is not a float, an int or a dolfinx.Constant
    """
    if isinstance(value, float | int):
        return fem.Constant(mesh, dolfinx.default_scalar_type(float(value)))
    elif isinstance(value, fem.Constant):
        return value
    else:
        raise TypeError(
            f"Value must be a float, an int or a dolfinx.Constant, not {type(value)}"
        )


def as_mapped_function(
    value: Callable,
    function_space: Optional[fem.functionspace] = None,
    t: Optional[fem.Constant] = None,
    temperature: Optional[fem.Function | fem.Constant | ufl.core.expr.Expr] = None,
) -> ufl.core.expr.Expr:
    """Maps a user given callable function to the mesh, time or temperature within
    festim as needed

    Args:
        value: the callable to convert
        function_space: the function space of the domain, optional
        t: the time, optional
        temperature: the temperature, optional

    Returns:
        The mapped function
    """

    # Extract the input variable names in the callable function `value`
    arguments = value.__code__.co_varnames

    kwargs = {}
    if "t" in arguments:
        kwargs["t"] = t
    if "x" in arguments:
        x = ufl.SpatialCoordinate(function_space.mesh)
        kwargs["x"] = x
    if "T" in arguments:
        kwargs["T"] = temperature

    return value(**kwargs)


def as_fenics_interp_expr_and_function(
    value: Callable,
    function_space: dolfinx.fem.function.FunctionSpace,
    t: Optional[fem.Constant] = None,
    temperature: Optional[fem.Function | fem.Constant | ufl.core.expr.Expr] = None,
) -> tuple[fem.Expression, fem.Function]:
    """Takes a user given callable function, maps the function to the mesh, time or
    temperature within festim as needed. Then creates the fenics interpolation
    expression and function objects

    Args:
        value: the callable to convert
        function_space: The function space to interpolate function over
        t: the time, optional
        temperature: the temperature, optional

    Returns:
        fenics interpolation expression, fenics function
    """

    mapped_function = as_mapped_function(
        value=value, function_space=function_space, t=t, temperature=temperature
    )

    fenics_interpolation_expression = fem.Expression(
        mapped_function,
        function_space.element.interpolation_points(),
    )

    fenics_object = fem.Function(function_space)
    fenics_object.interpolate(fenics_interpolation_expression)

    return fenics_interpolation_expression, fenics_object


class Value:
    """
    A class to handle input values from users and convert them to a relevent fenics
    object

    Args:
        input_value: The value of the user input

    Attributes:
        input_value : The value of the user input
        fenics_interpolation_expression : The expression of the user input that is used
            to update the `fenics_object`
        fenics_object : The value of the user input in fenics format

    """

    input_value: float | int | fem.Constant | np.ndarray | fem.Expression | ufl.core.expr.Expr | fem.Function

    ufl_expression: ufl.core.expr.Expr
    fenics_interpolation_expression: fem.Expression
    fenics_object: fem.Function | fem.Constant | ufl.core.expr.Expr

    def __init__(self, input_value):
        self.input_value = input_value

        self.ufl_expression = None
        self.fenics_interpolation_expression = None
        self.fenics_object = None

    def __repr__(self) -> str:
        return str(self.input_value)

    @property
    def input_value(self):
        return self._input_value

    @input_value.setter
    def input_value(self, value):
        if value is None:
            self._input_value = value
        elif isinstance(
            value,
            float
            | int
            | fem.Constant
            | np.ndarray
            | fem.Expression
            | ufl.core.expr.Expr
            | fem.Function,
        ):
            self._input_value = value
        elif callable(value):
            self._input_value = value
        else:
            raise TypeError(
                "Value must be a float, int, fem.Constant, np.ndarray, fem.Expression,"
                f" ufl.core.expr.Expr, fem.Function, or callable not {value}"
            )

    @property
    def time_dependent(self) -> bool:
        """Returns true if the value given is time dependent"""
        if self.input_value is None:
            return False
        if isinstance(self.input_value, fem.Constant | ufl.core.expr.Expr):
            return False
        if callable(self.input_value):
            arguments = self.input_value.__code__.co_varnames
            return "t" in arguments
        else:
            return False

    @property
    def temperature_dependent(self) -> bool:
        """Returns true if the value given is temperature dependent"""
        if self.input_value is None:
            return False
        if isinstance(self.input_value, fem.Constant | ufl.core.expr.Expr):
            return False
        if callable(self.input_value):
            arguments = self.input_value.__code__.co_varnames
            return "T" in arguments
        else:
            return False

    def convert_input_value(
        self,
        mesh: Optional[dolfinx.mesh.Mesh] = None,
        function_space: Optional[dolfinx.fem.function.FunctionSpace] = None,
        t: Optional[fem.Constant] = None,
        temperature: Optional[fem.Function | fem.Constant | ufl.core.expr.Expr] = None,
        up_to_ufl_expr: Optional[bool] = False,
    ):
        """Converts a user given value to a relevent fenics object depending
        on the type of the value provided

        Args:
            mesh: the mesh of the domain, optional
            function_space: the function space of the fenics object, optional
            t: the time, optional
            temperature: the temperature, optional
            up_to_ufl_expr: if True, the value is only mapped to a function if the input 
                is callable, not interpolated or converted to a function, optional
        """
        if isinstance(
            self.input_value, fem.Constant | fem.Function | ufl.core.expr.Expr
        ):
            self.fenics_object = self.input_value

        elif isinstance(self.input_value, fem.Expression):
            self.fenics_interpolation_expression = self.input_value

        elif isinstance(self.input_value, float | int):
            self.fenics_object = as_fenics_constant(value=self.input_value, mesh=mesh)

        elif callable(self.input_value):
            args = self.input_value.__code__.co_varnames
            # if only t is an argument, create constant object
            if "t" in args and "x" not in args and "T" not in args:
                if not isinstance(self.input_value(t=float(t)), float | int):
                    raise ValueError(
                        "self.value should return a float or an int, not "
                        + f"{type(self.input_value(t=float(t)))} "
                    )

                self.fenics_object = as_fenics_constant(
                    value=self.input_value(t=float(t)), mesh=mesh
                )

            elif up_to_ufl_expr:
                self.fenics_object = as_mapped_function(
                    value=self.input_value,
                    function_space=function_space,
                    t=t,
                    temperature=temperature,
                )

            else:
                self.fenics_interpolation_expression, self.fenics_object = (
                    as_fenics_interp_expr_and_function(
                        value=self.input_value,
                        function_space=function_space,
                        t=t,
                        temperature=temperature,
                    )
                )

    def update(self, t: float):
        """Updates the value

        Args:
            t: the time
        """
        if callable(self.input_value):
            arguments = self.input_value.__code__.co_varnames

            if isinstance(self.fenics_object, fem.Constant) and "t" in arguments:
                self.fenics_object.value = float(self.input_value(t=t))

            elif isinstance(self.fenics_object, fem.Function):
                if self.fenics_interpolation_expression is not None:
                    self.fenics_object.interpolate(self.fenics_interpolation_expression)
