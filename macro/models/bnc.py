import pandas as pd
from sympy import Symbol, Expr, Eq, solve
from typing import NamedTuple


eta = Symbol(r'\eta')
t = Symbol('t')
labor = Symbol('L')
c = Symbol('C')


class BncFunctions(NamedTuple):
    labor_function: Expr
    tfp: Expr
    consumption_utility: Expr
    labor_disutility: Expr


class Bnc:

    def __init__(self, functions):
        self.functions = functions
        self.eq_labor = self._calc_eq_labor()

    def _l_prime(self):
        return self.functions.labor_function.diff(labor)

    def _v_prime(self):
        consumption = (1 - eta) * self.functions.tfp * self.functions.labor_function
        return self.functions.consumption_utility.diff(c).subs(c, consumption)

    def _h_prime(self):
        return self.functions.labor_disutility.diff(labor)

    def _calc_eq_labor(self):
        h_t = self.functions.tfp * self._l_prime() * self._v_prime() + self._h_prime()
        return solve(Eq(h_t, 0), labor)[0]

    def _get_eq_functions(self, param_dict):
        output = self.functions.tfp * self.functions.labor_function.subs(labor, self.eq_labor)
        wage = self.functions.tfp * self.functions.labor_function.diff(labor).subs(labor, self.eq_labor)
        return output.subs(param_dict), wage.subs(param_dict)

    def simulate(self, periods, param_dict):
        history = pd.DataFrame(index=range(periods), columns=['y', 'w'])
        eq_functions = self._get_eq_functions(param_dict)
        for period in range(periods):
            history.loc[period] = [f.subs(t, period) for f in eq_functions]
        history['c'] = history['y'] * (1 - param_dict[eta])
        history['g'] = history['y'] * param_dict[eta]
        return history
