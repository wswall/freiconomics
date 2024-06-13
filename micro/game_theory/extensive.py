from copy import deepcopy
from collections import defaultdict
import os

import matplotlib.pyplot as plt
import networkx as nx


os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


def _get_node_history(node):
    history = []
    if node.parent_node is not None:
        history.append(node.parent_node)
        history.extend(_get_node_history(node.parent_node))
    return history


class Node:
    def __init__(self, player, parent_node=None, name=None, payout=None):
        self.player = player
        self.parent_node = parent_node
        self.name = name or id(self)
        self.payout = payout

    def update_payout(self, payout):
        self.payout = payout

    def get_history(self):
        return list(reversed(_get_node_history(self)))

    def is_terminal(self):
        return self.payout is not None

    def __repr__(self):
        return f"Node {self.name}"

    def __str__(self):
        return f"Node {self.name}"


class ExtensiveGame:
    def __init__(self, n_players, assign_id="len"):
        self.players = list(range(n_players))
        self.assign_id = assign_id
        self.tree = nx.Graph()
        self.tree.add_node(0, data=Node(1, name="root"))

    @property
    def root_node(self):
        return self.tree.nodes[0]["data"]

    def node_count(self):
        return len(self.tree.nodes)

    def _generate_node(self, player, parent_name, name=None, payout=None):
        parent_node = self.tree.nodes[parent_name]["data"]
        if name is None and self.assign_id == "len":
            name = self.node_count()
        return Node(player, parent_node, name=name, payout=payout)

    def add_node(self, player, parent_name, action, name=None, payout=None):
        new_node = self._generate_node(player, parent_name, name=name, payout=payout)
        self.tree.add_node(new_node.name, data=new_node)
        self.tree.add_edge(parent_name, new_node.name, action=action)
        return new_node

    def _get_node_list(self):
        if len(self.tree.nodes) == 1:
            return self.tree.nodes
        return [
            node
            for node, data in self.tree.nodes.items()
            if not data["data"].is_terminal()
        ]

    def _get_node_labels(self):
        labels = {}
        for node, data in self.tree.nodes.items():
            if data["data"].is_terminal():
                payout = data["data"].payout
                labels[node] = f"\n\n{payout}"
            else:
                labels[node] = str(data["data"].player)
        return labels

    def _get_edge_labels(self):
        return {x: y["action"] for x, y in self.tree.edges.items()}

    def _generate_pos(self):
        layout = nx.nx_pydot.pydot_layout(self.tree, prog="dot")
        return {int(x): y for x, y in layout.items()}

    def get_terminal_nodes(self):
        return [d['data'] for d in self.tree.nodes.values() if d['data'].is_terminal()]

    @staticmethod
    def _get_parent_payouts(terminal_nodes):
        parents = defaultdict(list)
        for node in terminal_nodes:
            parents[node.parent_node].append(node.payout)
        return parents

    def _update_parent_payouts(self, terminal_nodes):
        parent_payouts = self._get_parent_payouts(terminal_nodes)
        for parent, payouts in parent_payouts.items():
            parent.update_payout(max(payouts, key=lambda x: x[parent.player - 1]))

    def backwards_induction_solve(self):
        extensive_game = deepcopy(self)
        while len(extensive_game.tree.nodes) > 1:
            terminal_nodes = self.get_terminal_nodes()
            self._update_parent_payouts(terminal_nodes)
            extensive_game.tree.remove_nodes_from([node.name for node in terminal_nodes])
        return extensive_game.tree.nodes[0]['data'].payout

    def plot(self):
        plt.figure()
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
