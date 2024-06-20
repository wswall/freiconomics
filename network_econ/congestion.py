from sympy import Function, Eq, solve
from sympy.abc import omega, Q


k = Function('k')(Q, omega)
rho = Function(r'\rho')(omega)
P = Function('P')(Q)


class CongestionCost:

    def __init__(self, k_q=None, rho_omega=None, p_q=None):
        self.subs = {
            self.k: k_q or k,
            self.rho: rho_omega or rho,
            self.P: p_q or P
        }

    def congestion_externality(self):
        return Q * self.k.diff(Q)

    def social_marginal_cost(self):
        return self.k + self.congestion_externality()

    def calc_Q_optimal(self):
        foc = Eq((self.P * Q - self.k * Q - self.rho).subs(self.subs).diff(Q), 0)
        return solve(foc, Q)[0]

    def calc_omega_optimal(self):
        # Calculate profit-maximizing/socially optimal network capacity
        foc = Eq((self.P * Q - self.k * Q - self.rho).subs(self.subs).diff(omega), 0)
        return solve(foc.subs(Q, self.calc_Q_optimal()), omega)[0]
