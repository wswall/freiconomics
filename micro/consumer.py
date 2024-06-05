from __future__ import annotations

from typing import Tuple, Callable, Mapping

from sympy import lambdify, IndexedBase, Indexed, Symbol, solve, Expr
from sympy.abc import y, u


class UtilityFunction:
    """
    Represents a utility function.

    Args:
        function (sympy.Expr): The mathematical expression representing the utility function.
        unit_var (str, optional): The variable name used in the utility function. Defaults to "x".

    Attributes:
        expression (sympy.Expr): The mathematical expression representing the utility function.
        unit_vector (list): The list of unit vectors used in the utility function.
        _callable (function): The callable function generated from the utility function expression.

    Methods:
        __call__(*args): Evaluates the utility function with the given arguments.

    """

    def __init__(self, function, unit_var="x"):
        self.expression = function
        self.unit_vector = self._get_unit_vector(unit_var)
        self._callable = lambdify([unit_var], self.expression)

    def __call__(self, *args):
        return self._callable(*args)

    def _get_unit_vector(self, unit_var):
        return [
            symbol
            for symbol in self.expression.free_symbols
            if unit_var in str(symbol).lower() and isinstance(symbol, Indexed)
        ]


x = IndexedBase("x", positive=True)
p = IndexedBase("p", positive=True)
lambda_ = Symbol(r"\lambda")


class Consumer:
    """
    A class representing a consumer in an economic model.

    Attributes:
        utility (Utility): The utility function representing the consumer's preferences.
        marshallian_demand (Mapping): A dictionary of callable functions representing
            the Marshallian demand for each good.
        hicksian_demand (Mapping): A dictionary of callable functions representing the
            Hicksian demand for each good.
    """

    def __init__(self, utility):
        """
        Initializes a Consumer object.

        Args:
            utility (Utility): The utility function representing the consumer's preferences.
        """
        self.utility = utility
        self.marshallian_demand = self._generate_demand_callables(
            self.generate_marshallian_demand, [y, p]
        )
        self.hicksian_demand = self._generate_demand_callables(
            self.generate_marshallian_demand, [u, p]
        )

    @property
    def budget(self) -> Expr:
        """
        Calculates the consumer's budget constraint.

        Returns:
            Expr: The symbolic expression representing the consumer's budget constraint.
        """
        return y - sum(p[i] * unit for i, unit in enumerate(self.utility.unit_vector))

    def _generate_demand_callables(self, demand_function: Callable, *args) -> Mapping[Expr, Callable]:
        """
        Generates callable functions for symbolic expressions of demand/utility per good.

        Args:
            demand_function (Callable): The demand function to generate expressions for.
            *args: Additional arguments to pass to the demand function.

        Returns:
            Mapping: A dictionary of callable functions representing the demand/utility 
            expressions for each good.
        """
        demand = demand_function()
        return {
            symbol: lambdify([*args], demand[symbol]) for symbol in self.utility.unit_vector
        }

    def _optimize(self, objective_function, constraint=None) -> Tuple[Mapping[Expr, Expr]]:
        """
        Optimizes the given objective function subject to the given constraint.

        Args:
            objective_function: The objective function to optimize.
            constraint: The constraint function to apply (optional).

        Returns:
            The solutions to the optimization problem.
        """
        lagrangian = objective_function + lambda_ * (constraint or 0)
        partials_wrt_units = [lagrangian.diff(unit) for unit in self.utility.unit_vector]
        partial_wrt_lambda = lagrangian.diff(lambda_)
        exclude_vars = [y, *[p[i + 1] for i, _ in enumerate(self.utility.unit_vector)]]
        return solve([*partials_wrt_units, partial_wrt_lambda], exclude=exclude_vars)

    def generate_marshallian_demand(self) -> Mapping[Expr, Expr]:
        """
        Generates a symbolic expression for the Marshallian demand of the given good.

        Returns:
            Mapping[Expr, Expr]: The solutions to the Marshallian demand optimization problem
        """
        return self._optimize(self.utility.expression, constraint=self.budget)[0]

    def generate_hicksian_demand(self) -> Mapping[Expr, Expr]:
        """
        Generates a symbolic expression for the Hicksian demand of the given good.

        Returns:
            Mapping[Expr, Expr]: The solutions to the Hicksian demand optimization problem.
        """
        return self._optimize(self.budget - y, constraint=u - self.utility.expression)[0]

    def evaluate_utility(self, quantities) -> float:
        """
        Evaluates the utility function for the given quantities.

        Args:
            quantities: The quantities of goods.

        Returns:
            float: The utility value.
        """
        return self.utility(quantities)

    def evaluate_marshallian_demand(self, income, prices) -> Tuple[float]:
        """
        Evaluates the Marshallian demand for each good given the consumer's income and prices.

        Args:
            income: The consumer's income.
            prices: The prices of goods.

        Returns:
            Tuple[float]: The quantities demanded for each good.
        """
        prices.insert(0, 0)
        output = []
        for symbol in self.utility.unit_vector:
            demand = self.marshallian_demand[symbol]
            output.append(demand([income, prices]))
        return output

    def evaluate_hicksian_demand(self, utility, prices) -> Tuple[float]:
        """
        Evaluates the Hicksian demand for each good given the desired utility level and prices.

        Args:
            utility: The desired utility level.
            prices: The prices of goods.

        Returns:
            Tuple[float]: The quantities demanded for each good.
        """
        prices.insert(0, 0)
        output = []
        for symbol in self.utility.unit_vector:
            demand = self.hicksian_demand[symbol]
            output.append(demand([utility, prices]))
        return output
