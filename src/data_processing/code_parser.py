from typing import Literal

import pandas as pd
import networkx as nx

import torch
from torch_geometric.data import Data

from tree_sitter import Parser, Language, Node, Tree
import tree_sitter_cpp
import tree_sitter_python
import tree_sitter_java
import tree_sitter_c_sharp
import tree_sitter_c
import tree_sitter_php
import tree_sitter_javascript
import tree_sitter_go

SUPPORTED_LANGUAGES = Literal[
    "Python", "Java", "C++", "C#", "C", "PHP", "JavaScript", "Go"
]

LANGUAGE_MAP: dict[SUPPORTED_LANGUAGES, Language | None] = {
    "Python": Language(tree_sitter_python.language()),
    "Java": Language(tree_sitter_java.language()),
    "C++": Language(tree_sitter_cpp.language()),
    "C#": Language(tree_sitter_c_sharp.language()),
    "C": Language(tree_sitter_c.language()),
    "PHP": Language(tree_sitter_php.language_php()),
    "JavaScript": Language(tree_sitter_javascript.language()),
    "Go": Language(tree_sitter_go.language()),
}


def create_nx_graph(code: str, language: SUPPORTED_LANGUAGES) -> nx.DiGraph:
    tree = _parse_code(code, language)
    root_node = tree.root_node
    g = nx.DiGraph()
    stack: list[tuple[Node, None | int]] = [(root_node, None)]
    node_id = 0

    while stack:
        node, parent_id = stack.pop()
        this_id = node_id
        node_id += 1

        snippet = code[node.start_byte : node.end_byte]
        snippet = snippet.replace("\n", "\\n")

        g.add_node(
            this_id,
            type=node.type,
            text=snippet.strip()[:40],
            start=node.start_point,
            end=node.end_point,
        )

        if parent_id is not None:
            g.add_edge(parent_id, this_id)

        for child in reversed(node.children):
            stack.append((child, this_id))

    return g


def _parse_code(code: str, language: SUPPORTED_LANGUAGES) -> Tree:
    parser = Parser(LANGUAGE_MAP[language])
    return parser.parse(bytes(code, "utf8"))


def create_graph(code: str, language: SUPPORTED_LANGUAGES) -> Data:
    tree = _parse_code(code, language)
    nodes, edges = [], []
    info = {
        "depth": [],
        "num_children": [],
        "type_id": [],
        "is_named": [],
        "length": [],
    }

    stack: list[tuple[Node, int | None, int]] = [(tree.root_node, None, 0)]

    while stack:
        node, parent_gid, depth = stack.pop(0)
        gid = len(nodes)

        nodes.append(node)
        info["depth"].append(depth)
        info["num_children"].append(len(node.children))
        info["type_id"].append(node.kind_id)
        info["is_named"].append(node.is_named)
        info["length"].append(node.end_byte - node.start_byte)

        if parent_gid is not None:
            edges.append((parent_gid, gid))
            edges.append((gid, parent_gid))

        for child in node.children:
            stack.append((child, gid, depth + 1))

    return Data(
        x=torch.stack(
            [
                torch.tensor(info["type_id"], dtype=torch.long),
                torch.tensor(info["depth"], dtype=torch.long),
                torch.tensor(info["num_children"], dtype=torch.long),
                torch.tensor(info["is_named"], dtype=torch.long),
                torch.tensor(info["length"], dtype=torch.long),
            ],
            dim=-1,
        ),
        edge_index=torch.tensor(edges, dtype=torch.long),
    )


def parse_data_frame(data: pd.DataFrame) -> list[Data]:
    parsed_codes = []
    for _, row in data.iterrows():
        parsed_codes.append(create_graph(row["code"], row["language"]))
    return parsed_codes


if __name__ == "__main__":
    from .data_loading import load_data

    new_data = load_data("b", "train").sample(1)
    print(parse_data_frame(new_data)[0])
