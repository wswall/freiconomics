from __future__ import annotations

from copy import deepcopy
from itertools import permutations
from string import ascii_letters, digits
from typing import Sequence, Mapping, List

import numpy as np
import pandas as pd


class Strategy:
    """
    Represents a player's strategy in a game.

    Attributes:
        name (str): The name of the strategy.
        payouts (Sequence[int|float]): The payouts associated with the
            strategy.
    """

    def __init__(self, name: str, payouts: Sequence[int | float]):
        self.name = name
        self.payouts = payouts

    def dominated_by(self, other_strategy: Strategy, strict: bool = False) -> bool:
        """
        Checks if the strategy is dominated by another strategy.

        Args:
            other_strategy (Strategy): The other strategy to compare
                against.
            strict (bool, optional): If True, check if the strategy is
                strictly dominated. If False, check if the strategy is
                weakly dominated. Defaults to False.

        Returns:
            bool: True if the strategy is dominated, False otherwise.
        """
        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean")

        payouts = self.payouts, other_strategy.payouts
        if strict:
            return all(p1 < p2 for p1, p2 in zip(*payouts))
        return all(p1 <= p2 for p1, p2 in zip(*payouts)) and any(
            p1 < p2 for p1, p2 in zip(*payouts)
        )


class Player:
    """
    Represents a player in a game.

    Attributes:
        name (str): The name of the player.
        strategies (Sequence[Strategy]): The strategies available to
            the player.
    """

    def __init__(self, name: str, strategies: Sequence[Strategy]):
        self.name = name
        self.strategies = strategies

    def update_strategies(self, new_strategies: Sequence[Strategy]):
        """
        Updates the strategies of the player.

        Args:
            new_strategies (Sequence[Strategy]): The new strategies to
                be assigned to the player.
        """
        self.strategies = new_strategies

    def dominated_strategies(self, strict: bool = False) -> set[str]:
        """
        Finds the dominated strategies of the player.

        Args:
            strict (bool, optional): If True, only considers strictly
                dominated strategies. If False, considers both strictly
                and weakly dominated strategies. Defaults to False.

        Returns:
            set[str]: A set of names of the dominated strategies.
        """
        if not isinstance(strict, bool):
            raise TypeError
        dominated_strategies = set()
        for strat1, strat2 in permutations(self.strategies, r=2):
            if strat1.dominated_by(strat2, strict=strict):
                dominated_strategies.add(strat1.name)
        return dominated_strategies


class StrategicGame:
    """
    Represents a game with players and strategies.

    Attributes:
        payouts (DataFrame): The payout matrix for the game.
        players (list[Player]): The list of players in the game.

    Methods:
        get_payouts: Get the payouts for a specific strategy.
        get_strategies: Get the list of strategies for a player.
        dominated_strategies: Get the set of dominated strategies.
        drop_strategy: Drop a strategy from the game.
    """

    def __init__(self, payout_df: pd.DataFrame):
        self.payouts = payout_df
        self.players = [
            Player(1, self.get_strategies()),
            Player(2, self.get_strategies(axis=1)),
        ]

    def get_payouts(self, strategy: str, axis: int = 0) -> Sequence[int | float]:
        """
        Get the payouts for a specific strategy.

        Args:
            strategy (str): The name of the strategy.
            axis (int, optional): The axis to consider for payouts.
                Defaults to 0.

        Returns:
            Sequence[int|float]: The payouts for the specified strategy
        """
        if axis == 1:
            return [x[axis] for x in self.payouts[strategy]]
        return [x[axis] for x in self.payouts.loc[strategy]]

    def get_strategies(self, axis: int = 0) -> List[Strategy]:
        """
        Get the list of strategies for a player.

        Args:
            axis (int, optional): The axis to consider for strategies.
                Defaults to 0.

        Returns:
            list[Strategy]: The list of strategies.
        """
        strategies = self.payouts.columns if axis == 1 else self.payouts.index
        return [Strategy(s, self.get_payouts(s, axis=axis)) for s in strategies]

    def _update_strategies(self) -> None:
        """Update the strategies for both players from the payout_df."""
        self.players[0].update_strategies(self.get_strategies())
        self.players[1].update_strategies(self.get_strategies(axis=1))

    @property
    def dominated_strategies(self) -> set[str]:
        """Get the joint set of dominated strategies form both players"""
        p1, p2 = self.players
        return p1.dominated_strategies().union(p2.dominated_strategies())

    def drop_strategy(self, strategy_name):
        """Drop a strategy from the game's payout_df and update player strategies"""
        if strategy_name in self.payouts.index:
            self.payouts.drop(strategy_name, inplace=True)
        elif strategy_name in self.payouts.columns:
            self.payouts.drop(strategy_name, axis=1, inplace=True)
        self._update_strategies()


def generate_game(low: int, high: int, size: tuple[int, int, int]) -> StrategicGame:
    """
    Generate a game with random payout matrix.

    Parameters:
        low (int): The lower bound for the random values in the payout
            matrix.
        high (int): The upper bound (exclusive) for the random values
            in the payout matrix.
        size (tuple[int, int, int]): The size of the payout matrix in
            the form (rows, columns, depth).

    Returns:
        Game: The generated game object.
    """
    payout_matrix = np.random.randint(low, high=high, size=size)
    df = pd.DataFrame(
        payout_matrix.tolist(),
        index=list(digits[:payout_matrix.shape[0]]),
        columns=list(ascii_letters[:payout_matrix.shape[1]])
    )
    return StrategicGame(df)


EliminationResults = List[tuple[StrategicGame, List[str]]]
ResultDict = Mapping[tuple[str], Mapping[str, List[str]]]


def prep_result_dict(elimination_results: EliminationResults) -> ResultDict:
    """
    Prepares a dictionary of results from the elimination process.

    Args:
        elimination_results (EliminationResults): The results of the
            elimination process.

    Returns:
        ResultDict: A dictionary containing the results of the
            elimination process. The keys are tuples representing the
            sequence of iterative elimination, and the values are
            dictionaries containing the surviving strategies of each
            player.

    """
    result_dict = {}
    for x, y in elimination_results:
        result_dict[tuple(y)] = {
            "player_1_strategies": x.payouts.index.tolist(),
            "player_2_strategies": x.payouts.columns.tolist(),
        }
    return result_dict


def iterative_elimination(
    game: StrategicGame, elimination_path: List[str] | None = None
) -> EliminationResults:
    """
    Performs iterative elimination of dominated strategies in a game.

    Args:
        game (Game): The game object representing the game to be analyzed.
        elimination_path (list|None, optional): The list of strategies
            eliminated so far. Defaults to None.

    Returns:
        EliminationResults: A list of tuples, where each tuple contains the
            game object after elimination and the elimination path.
    """
    results = []
    if elimination_path is None:
        elimination_path = []

    if not game.dominated_strategies:
        return [[game, elimination_path]]

    for d in game.dominated_strategies:
        game_copy = deepcopy(game)
        game_copy.drop_strategy(d)
        new_path = elimination_path + [d]
        results.extend(iterative_elimination(game_copy, elimination_path=new_path))
    return results
