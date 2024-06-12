from __future__ import annotations

from itertools import product
from typing import Sequence, Mapping
from numbers import Number

import numpy as np
from sympy import latex, solve, Symbol, Matrix, Eq, Integral, Indexed
from sympy.core.expr import Expr


p_tilde, p_bar = Symbol(r"\tilde p"), Symbol(r"\bar p")


def _get_var(function: Expr, var: str) -> Symbol:
    var_list = [
        symbol for symbol in function.free_symbols if var in str(symbol).lower()
    ]
    if len(var_list) == 1:
        return var_list[0]
    indexed_vars = [x for x in var_list if isinstance(x, Indexed)]
    if len(indexed_vars) == 1:
        return indexed_vars[0]
    raise ValueError


class DemandFunction:
    """
    Represent the demand function for a single good.

    Stores a sympy expression representing the demand function with convenience methods
    for solving in terms of price or quantity and calculating consumer surplus. By default,
    the function passed is assumed to define quantity as a function of price, using the 
    symbols q and p, respectively. It is further assumed that these symbols are present only
    once each. If other symbols are used or more than 1 instance of p or q is used, the
    symbol to solve for must be passed explicitly. If the demand function defines price as
    a function of quantity, the inverse flag should be set to True.

    Args:
        function (Expr): The mathematical expression representing the demand function.
        q_var (Symbol | Indexed, optional): The symbol or indexed variable representing
            the quantity variable. Defaults to None.
        p_var (Symbol | Indexed, optional): The symbol or indexed variable representing
            the price variable. Defaults to None.
        inverse (bool, optional): Indicates whether the demand function is an inverse
            demand function. Defaults to False.

    Attributes:
        function: The mathematical expression representing the demand function.
        inverse (bool): Indicates whether the demand function is an inverse demand
            function.
        vars (dict): A dictionary containing the quantity and price variables.
    """

    def __init__(
        self,
        function: Expr,
        q_var: Symbol | Indexed = None,
        p_var: Symbol | Indexed = None,
        inverse: bool = False,
    ):
        self.function = function
        self.inverse = inverse
        self.vars = {
            "q": q_var or _get_var(function, "q"),
            "p": p_var or _get_var(function, "p"),
        }

    def __getitem__(self, item) -> Symbol | Indexed:
        return self.vars[item]

    def __repr__(self):
        return str(self.function)

    def _repr_latex_(self):
        return latex(self.function)

    def p_q(self) -> Expr:
        """
        Returns the equation as p(q) = function.

        If the 'inverse' flag is set to True, the original function is returned.
        Otherwise, the equation q = function(p) is solved for p and the result is
        returned.

        Returns:
            The value of p that satisfies the equation q = function(p).
        """
        if self.inverse:
            return self.function
        return solve(Eq(self["q"], self.function), self["p"])[0]

    def q_p(self) -> Expr:
        """
        Returns the equation as q(p) = function.

        If the 'inverse' flag is set to False, the original function is returned.
        Otherwise, the equation p = function(q) is solved for q and the result is
        returned.

        Returns:
            The value of q that satisfies the equation p = function(q).
        """
        if not self.inverse:
            return self.function
        return solve(Eq(self["p"], self.function), self["q"])[0]

    def area_under_curve(self) -> Integral:
        """
        Return symbolic representation of area under the curve between p and p_bar.

        Returns:
            Integral: The integral representing the area under the curve.
        """
        return Integral(
            self.q_p().subs(self["p"], p_tilde), (p_tilde, self["p"], p_bar)
        )


def is_maximum(hessian: Matrix) -> bool:
    """Checks if the given Hessian matrix represents a maximum point"""
    return all(val < 0 for val in hessian.eigenvals())


class Industry:
    """
    Represents an industry with demand functions and a cost function.

    Attributes:
        demand_functions (Sequence[DemandFunction]): A sequence of demand functions.
        cost_function (Expr): The cost function of the industry.

    Args:
        demand_functions (Sequence[DemandFunction]): A sequence of demand functions.
        cost_function (Expr): The cost function.
    """

    _lambda = Symbol(r"\lambda")

    def __init__(self, demand_functions: Sequence[DemandFunction], cost_function: Expr):
        self.demand_functions = demand_functions
        self.cost_function = cost_function

    def revenue(self) -> Expr:
        """Returns expression for total revenue of the industry."""
        return sum(demand["p"] * demand.q_p() for demand in self.demand_functions)

    def consumer_surplus(self) -> Expr:
        """Returns expression for total consumer surplus of the industry."""
        return sum(demand.area_under_curve() for demand in self.demand_functions)

    def profit(self) -> Expr:
        """Returns expression for the profit of the industry."""
        return self.revenue() - self.cost_function

    def generate_hessian(self, lagrangian: Expr, solution) -> Matrix:
        """
        Generates the Hessian matrix for the given lagrangian and solution.

        Args:
            lagrangian (Expr): Symbolic expression of the Lagrangian equation.
            solution: A solution to the optimization problem.

        Returns:
            Matrix: The Hessian matrix.
        """
        # Get each i,j tuple for second order differentiation
        index_range = range(len(self.demand_functions))
        second_diff_pairs = product(list(index_range), repeat=2)
        # Iterate through pairs, calculating second order diffs and storing in matrix
        hessian = np.zeros((len(self.demand_functions), len(self.demand_functions)))
        for pair in list(second_diff_pairs):
            p1 = self.demand_functions[pair[0]]["p"]
            p2 = self.demand_functions[pair[1]]["p"]
            second_order_diff = lagrangian.diff(p1).diff(p2)
            hessian[pair[0], pair[1]] = second_order_diff.subs(
                self._lambda, solution[self._lambda]
            )
        return Matrix(hessian)

    def _get_optimization_solutions(self, lagrangian: Expr) -> Mapping[Symbol, Number]:
        partials_wrt_prices = [
            lagrangian.diff(demand["p"]) for demand in self.demand_functions
        ]
        partial_wrt_lambda = lagrangian.diff(self._lambda)
        return solve([*partials_wrt_prices, partial_wrt_lambda])

    def get_ramsey_prices(self) -> Sequence[Mapping[Symbol, Number]]:
        """
        Finds the Ramsey prices for the industry.

        Returns:
            Sequence[Mapping[Symbol, Number]]: A sequence of mappings representing the
                Ramsey prices.
        """
        lagrangian = (
            self.consumer_surplus() + self.profit() - self._lambda * self.profit()
        )
        return [
            solution
            for solution in self._get_optimization_solutions(lagrangian)
            if is_maximum(self.generate_hessian(lagrangian, solution))
        ]
