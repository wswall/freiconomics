from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence, Mapping, Tuple, Any, Generator, Hashable

import matplotlib.pyplot as plt
import networkx as nx


os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

StrategySequence = Sequence[Tuple[Any, Any]]


@dataclass
class Node:
    name: str | int
    history: list[str]
    player: str | int | None = None
    payout: list[int] | None = None


class Strategy:

    def __init__(self, sequence: Sequence[Tuple[Any, Any]], payout: Sequence[int]):
        self.nodes = list(self._generate_nodes(sequence, payout))

    def _generate_nodes(
        self, sequence: StrategySequence, payout: Sequence[int]
    ) -> Generator[Node, None, None]:
        history = ["x0"]
        for i, (_, action) in enumerate(sequence):
            history = [*history, action]
            node_name = "".join(history[1:])
            if i < len(sequence) - 1:
                yield action, Node(node_name, history, player=sequence[i + 1][0])
            else:
                yield action, Node(node_name, history, payout=payout)

    def __iter__(self) -> Generator[Node, None, None]:
        for node in self.nodes:
            yield node


class ExtensiveGame:
    def __init__(self):
        self.tree = nx.Graph()
        self._nodes = {}

    def __getitem__(self, node_name: Hashable):
        node_idx = self._nodes[node_name]
        return self.tree.nodes[node_idx]

    @property
    def root_node(self):
        """Fetch the root node of the game tree"""
        return self.tree.nodes[0]

    def node_count(self):
        """Get a count of the nodes in the game tree"""
        return len(self.tree.nodes)

    def set_root(self, player: str | int) -> None:
        """Create the root node of the game tree by specifying the starting player"""
        node = Node("root", ["x0"], player=player)
        self.tree.add_node(0, data=node)
        self._nodes[node.name] = 0

    @staticmethod
    def _is_terminal(node: Node) -> bool:
        return node.payout is not None

    def _get_node_list(self) -> Sequence[str]:
        if len(self.tree.nodes) == 1:
            return self.tree.nodes
        return [
            node_name
            for node_name, node in self.tree.nodes.items()
            if not self._is_terminal(node["data"])
        ]

    def _get_node_labels(self) -> Mapping[int, str]:
        labels = {}
        for node_idx, node in self.tree.nodes.items():
            node_data = node["data"]
            if self._is_terminal(node_data):
                labels[node_idx] = f"\n\n{node_data.payout}"
            else:
                labels[node_idx] = str(node_data.player)
        return labels

    def _get_edge_labels(self) -> Mapping[Tuple[int, int], Any]:
        return {x: y["action"] for x, y in self.tree.edges.items()}

    def _generate_pos(self) -> Mapping[int, Tuple[float, float]]:
        layout = nx.nx_pydot.pydot_layout(self.tree, prog="dot")
        return {int(x): y for x, y in layout.items()}

    def get_terminal_nodes(self) -> Sequence[Node]:
        """Get a list of terminal nodes in the game tree"""
        return [
            node for node in self.tree.nodes.values() if self._is_terminal(node["data"])
        ]

    def _add_action(self, parent_node_index: int, node_index: int, action: Any) -> None:
        if not self.tree.has_edge(parent_node_index, node_index):
            self.tree.add_edge(parent_node_index, node_index, action=action)

    def add_terminal_node(
        self, strategy: StrategySequence, payout: Sequence[Any]
    ) -> None:
        """
        Add a terminal node to the game tree.

        Takes a list of player, action tuples and the payout associated
        with that set of actions. For each action, if the node does not
        exist in the tree, a new node is created and a new edge is
        created linking it to its parent node.

        Args:
            strategy (StrategySequence): A sequence of player, action 
                tuples representing a complete game.
            payout (Sequence[Any]): The payout associated with the 
                terminal node.

        Returns:
            None
        """
        last_node = 0
        strategy = Strategy(strategy, payout)
        for action, node in strategy:
            node_idx = self.node_count()
            if node.name not in self._nodes:
                self._nodes[node.name] = node_idx
                self.tree.add_node(node_idx, data=node)
                self._add_action(last_node, node_idx, action)
            else:
                node_idx = self._nodes[node.name]
            last_node = node_idx

    def add_terminal_nodes(
        self,
        strategies_and_payouts: Sequence[Sequence[StrategySequence, Sequence[Any]]],
    ) -> None:
        """
        Add multiple terminal nodes to the game tree.

        Takes list of lists where each list consists of player, action
        tuples and the payout associated with that set of actions. For
        each action, if the node does not exist in the tree, a new node
        is created and a new edge is created linking it to its parent
        node.

        Args:
            strategy (StrategySequence): A sequence of player, action 
                tuples representing a complete game.
            payout (Sequence[Any]): The payout associated with the 
                terminal node.

        Returns:
            None
        """
        for strategy, payout in strategies_and_payouts:
            self.add_terminal_node(strategy, payout)

    def plot(self, figsize: Tuple[int, int] = (10, 10)) -> None:
        """
        Plot the game tree.

        Args:
            figsize (Tuple[int, int], optional): The size of the figure
                Defaults to (10, 10).
        """
        plt.figure(figsize=figsize)
        pos = self._generate_pos()
        nx.draw_networkx_nodes(
            self.tree, pos, nodelist=self._get_node_list(), node_color="#679ec7"
        )
        nx.draw_networkx_edges(self.tree, pos)
        nx.draw_networkx_edge_labels(
            self.tree, pos, edge_labels=self._get_edge_labels()
        )
        nx.draw_networkx_labels(self.tree, pos, labels=self._get_node_labels())
        plt.axis("off")
        plt.show()
