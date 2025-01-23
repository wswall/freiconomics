from collections import namedtuple

import pandas as pd


Params = namedtuple(
    "Params", ["eta", "tau", "pi_target", "phi_pi", "phi_b", "g", "beta"]
)


class MonetaryDynamics:

    def __init__(self, params):
        self.params = params
        self.r_star = (1 + self.params.g) / self.params.beta - 1
        self.policy = self._check_policy()
        self.steady_state = self._calc_steady_state()

    def _check_params(self):
        if not 0 < self.params.beta < 1:
            raise ValueError("parameter beta must be between 0 and 1")
        if self.params.pi_target <= self.params.beta - 1:
            raise ValueError("parameter pi_target must be greater than beta - 1")
        if self.params.phi_pi / (1 + self.r_star) == 1:
            raise ValueError("phi_pi / (1 + r) must be different from 1")
        if self.r_star - self.params.phi_b - self.params.g == 0:
            raise ValueError("r - phi_b - g must be different from 0")

    def monetary_active(self):
        return self.params.phi_pi > 1 + self.r_star

    def fiscal_active(self):
        return self.params.phi_b < self.r_star - self.params.g

    def _check_policy(self):
        policy = self.monetary_active() - self.fiscal_active()
        if policy == 0:
            raise ValueError(
                "Invalid parameter combination phi_pi, phi_b, g, and beta. \
                Equilibrium indeterminate or does not exist under this configuration."
            )
        return "Ricardian" if policy == 1 else "Non-Ricardian"

    def _calc_steady_state(self):
        b_bar = (
            (self.params.eta - self.params.tau)
            * (1 + self.params.g)
            / (self.params.g + self.params.phi_b - self.r_star)
        )
        return {
            "pi": self.params.pi_target,
            "b": b_bar,
            "i": (1 + self.r_star) * (1 + self.params.pi_target) - 1,
            "s_f": (
                self.params.tau
                - self.params.eta
                + self.params.phi_b / (1 + self.params.g) * b_bar
            ),
        }

    def calc_pi_0(self, b_init, i_init):
        if self.policy == "Ricardian":
            return self.steady_state["pi"]
        divisor = (self.steady_state["b"] + self.params.tau - self.params.eta) * (
            1 + self.params.g
        ) + self.params.phi_b * b_init
        return (1 + i_init) * b_init / divisor - 1

    def calc_b_0(self, b_init, i_init):
        if self.policy != "Ricardian":
            return self.steady_state["b"]
        r_0 = (1 + i_init) / (1 + self.params.pi_target) - 1
        return (
            self.params.eta
            - self.params.tau
            + (1 + r_0 - self.params.phi_b) / (1 + self.params.g) * b_init
        )

    def calc_y(self, y):
        return y * (1 + self.params.g)

    def calc_pi_t(self, pi):
        return self.params.pi_target + self.params.phi_pi / (1 + self.r_star) * (
            pi - self.params.pi_target
        )

    def calc_b_t(self, b):
        return (
            self.params.eta
            - self.params.tau
            + (1 + self.r_star - self.params.phi_b) / (1 + self.params.g) * b
        )

    def calc_i_series(self, pi_series):
        return (
            (1 + self.r_star) * (1 + self.params.pi_target)
            - 1
            + self.params.phi_pi * (pi_series - self.params.pi_target)
        )

    def calc_T_series(self, Y_series, B_series, B_init):
        B_series = pd.concat(B_init, B_series)[:-1]
        return self.params.tau * Y_series + self.params.phi_b * B_series

    def simulate(self, periods, i_init=0.05, b_init=1.18, y_init=67632):
        history = pd.DataFrame(index=range(periods), columns=["Y", "pi", "b", "B"])
        y = self.calc_y(y_init)
        pi, b = self.calc_pi_0(b_init, i_init), self.calc_b_0(b_init, i_init)
        for period in range(periods):
            history.loc[period] = [y, pi, b, b * y]
            y = self.calc_y(y)
            pi, b = self.calc_pi_t(pi), self.calc_b_t(b)

        history["i"] = self.calc_i_series(history["pi"])
        history["B"] = history["b"] * history["Y"]
        history["G"] = history["Y"] * self.params.eta
        history.loc["T"] = self.calc_T_series(
            history["Y"], history["B"], b_init * y_init
        )
        history["s_f"] = (history["T"] - history["G"]) / history["Y"]

        return history
