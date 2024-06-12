from __future__ import annotations

from typing import Tuple, Callable, Mapping, Sequence

from sympy import lambdify, IndexedBase, Indexed, Symbol, solve, Expr
from sympy.abc import y, u


class UtilityFunction:
    """
    Represents a utility function.

    Args:`
        function (sympy.Expr): The symbolic expression representing the
            utility function.
        unit_var (str, optional): The variable name used to represent
            quantity in the utility function. Defaults to "x".

    Attributes:
        expression (sympy.Expr): The symbolic expression representing
            the utility function.
        quantity_vector (list): The list of unit vectors used in the
            utility function.
        _callable (function): The callable function generated from the
            utility function expression.

    Methods:
        __call__(*args): Evaluates the utility function with the given
            arguments.
    """

    def __init__(self, function: Expr, unit_var: str = "x"):
        self.expression = function
        self.quantity_vector = self._get_quantity_vector(unit_var)
        self._callable = lambdify([unit_var], self.expression)

    def __call__(self, *args) -> float:
        return self._callable(*args)

    def _get_quantity_vector(self, unit_var: str) -> Sequence[Expr]:
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
        utility (Utility): The utility function representing the
            consumer's preferences.
        marshallian_demand (Mapping): A dictionary of callable
            functions representing the Marshallian demand for each
            good.
        hicksian_demand (Mapping): A dictionary of callable functions
            representing the Hicksian demand for each good.
    """

    def __init__(self, utility: Expr):
        self.utility = utility
        self.marshallian_demand = self._generate_demand_callables(
            self.generate_marshallian_demand, [y, p]
        )
        self.hicksian_demand = self._generate_demand_callables(
            self.generate_marshallian_demand, [u, p]
        )

    @property
    def q_vec(self):
        return self.utility.quantity_vector

    @property
    def budget(self) -> Expr:
        """
        Calculates the consumer's budget constraint.

        Returns:
            Expr: The symbolic expression representing the consumer's
                budget constraint.
        """
        return y - sum(p[i] * unit for i, unit in enumerate(self.q_vec))

    def _generate_demand_callables(
        self, demand_function: Callable, *args
    ) -> Mapping[Expr, Callable]:
        """
        Generates callables for symbolic expressions of demand good.

        Args:
            demand_function (Callable): The demand function to generate
                expressions for.
            *args: Additional arguments to pass to the demand function.

        Returns:
            Mapping: A dictionary of callable functions representing
                the demand/utility expressions for each good.
        """
        demand = demand_function()
        return {
            symbol: lambdify([*args], demand[symbol])
            for symbol in self.q_vec
        }

    def _optimize(
        self, objective_function: Expr, constraint: Expr | None = None
    ) -> Tuple[Mapping[Expr, Expr]]:
        """
        Optimize the objective function subject to the constraint.

        Args:
            objective_function: The objective function to optimize.
            constraint: The constraint function to apply (optional).

        Returns:
            The solutions to the optimization problem.
        """
        lagrangian = objective_function + lambda_ * (constraint or 0)
        partials = [
            *[lagrangian.diff(unit) for unit in self.q_vec],
            lagrangian.diff(lambda_)
        ]
        price_vars = [p[i + 1] for i, _ in enumerate(self.q_vec)]
        return solve(partials, exclude=[y, *price_vars])

    def generate_marshallian_demand(self) -> Mapping[Expr, Expr]:
        """
        Generates symbolic expression for the Marshallian demand.

        Returns:
            Mapping[Expr, Expr]: The solutions to the Marshallian
                demand optimization problem
        """
        return self._optimize(self.utility.expression, constraint=self.budget)[0]

    def generate_hicksian_demand(self) -> Mapping[Expr, Expr]:
        """
        Generates a symbolic expression for the Hicksian demand.

        Returns:
            Mapping[Expr, Expr]: The solutions to the Hicksian demand
                optimization problem.
        """
        objective, constraint = self.budget - y, u - self.utility.expression
        return self._optimize(objective, constraint=constraint)[0]

    def evaluate_utility(self, quantities: Sequence[int | float]) -> float:
        """
        Evaluates the utility function for the given quantities.

        Args:
            quantities(Sequence[int | float]): The quantities of goods.

        Returns:
            float: The utility value.
        """
        return self.utility(quantities)

    def _evaluate_demand(self, demand_type, *args):
        output = []
        for symbol in self.q_vec:
            demand = demand_type[symbol]
            output.append(demand(*args))
        return output

    def evaluate_marshallian_demand(
        self, income: int | float, prices: Sequence[int | float]
    ) -> Tuple[float]:
        """
        Evaluates demand for each good given consumer's income and prices.

        Args:
            income: The consumer's income.
            prices: The prices of goods.

        Returns:
            Tuple[float]: The quantities demanded for each good.
        """
        # Lambdified demand function will use the 0th element of prices as p[0], need
        # to pad prices are inserted correctly
        prices.insert(0, 0)
        return self._evaluate_demand(self.marshallian_demand, income, prices)

    def evaluate_hicksian_demand(
        self, utility: int | float, prices: Sequence[int | float]
    ) -> Tuple[float]:
        """
        Evaluate demand for each good given utility level and prices.

        Args:
            utility: The desired utility level.
            prices: The prices of goods.

        Returns:
            Tuple[float]: The quantities demanded for each good.
        """
        # Lambdified demand function will use the 0th element of prices as p[0], need
        # to pad prices are inserted correctly
        prices.insert(0, 0)
        return self._evaluate_demand(self.marshallian_demand, utility, prices)
