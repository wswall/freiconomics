import pandas as pd
import random
from sympy import Symbol, Expr, Eq, solve
from typing import NamedTuple


eta = Symbol(r'\eta')
t = Symbol('t')
labor = Symbol('L')
c = Symbol('C')


def generate_param_sequence(values, transitions, samples, initial_state=None):
    states = list(transitions)
    path = []
    state = initial_state or states[0]
    for _ in range(samples):
        state = random.choices(states, weights=transitions[state])[0]
        path.append(state)
    return [values[state] for state in path]


def to_list_of_dicts(dict_of_lists):
    lengths = [len(v) for v in dict_of_lists.values()]
    if not len(set(lengths)) == 1:
        raise ValueError("All lists must have the same length.")
    return [{key: dict_of_lists[key][i] for key in dict_of_lists} for i in range(lengths[0])]


def stochastics_to_sequences(stochastics_dict, length):
    output = {}
    for param in stochastics_dict:
        values, weights = stochastics_dict[param]["values"], stochastics_dict[param]["weights"]
        output[param] = generate_param_sequence(values, weights, length)
    return to_list_of_dicts(output)


class ModelFunctions(NamedTuple):
    labor_function: Expr
    tfp: Expr
    consumption_utility: Expr
    labor_disutility: Expr


class NewClassicalModel:

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
        # fails to solve without simplifying first
        return solve(Eq(h_t.simplify(), 0), labor)[0]

    def _get_eq_functions(self, param_dict):
        output = self.functions.tfp * self.functions.labor_function.subs(labor, self.eq_labor)
        wage = self.functions.tfp * self.functions.labor_function.diff(labor).subs(labor, self.eq_labor)
        return output.subs(param_dict), wage.subs(param_dict)

    def _simulate_constant(self, periods, param_dict):
        history = pd.DataFrame(index=range(periods), columns=['y', 'w'])
        eq_functions = self._get_eq_functions(param_dict)
        for period in range(periods):
            history.loc[period] = [f.subs(t, period) for f in eq_functions]
        return history

    def _simulate_stochastic(self, periods, param_dict, param_sequences):
        history = pd.DataFrame(index=range(periods), columns=['y', 'w'])
        if isinstance(param_sequences, dict):
            param_sequences = to_list_of_dicts(param_sequences)
        for period in range(periods):
            param_dict.update(param_sequences[period])
            eq_functions = self._get_eq_functions(param_dict)
            history.loc[period] = [f.subs(t, period) for f in eq_functions]
        return history

    def simulate(self, periods, param_dict, param_sequences=None):
        if param_sequences is None:
            history = self._simulate_constant(periods, param_dict)
        else:
            history = self._simulate_stochastic(periods, param_dict, param_sequences)
        history['c'] = history['y'] * (1 - param_dict[eta])
        history['g'] = history['y'] * param_dict[eta]
        return history
