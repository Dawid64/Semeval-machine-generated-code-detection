from pathlib import Path
from typing import Literal

import pandas as pd
import networkx as nx

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


def _tree_to_graph(tree: Tree, code_str: str) -> nx.DiGraph:
    root_node = tree.root_node
    g = nx.DiGraph()
    stack: list[tuple[Node, None | int]] = [(root_node, None)]
    node_id = 0

    while stack:
        node, parent_id = stack.pop()
        this_id = node_id
        node_id += 1

        snippet = code_str[node.start_byte : node.end_byte]
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


def parse_code(code: str, language: SUPPORTED_LANGUAGES) -> nx.DiGraph:
    parser = Parser(LANGUAGE_MAP[language])
    tree = parser.parse(bytes(code, "utf8"))
    return _tree_to_graph(tree, code)


def parse_data_frame(data: pd.DataFrame) -> list[nx.DiGraph]:
    parsed_codes = []
    for _, row in data.iterrows():
        graph = parse_code(row["code"], row["language"])
        parsed_codes.append(graph)
    return parsed_codes


if __name__ == "__main__":
    path = Path("SemEval-2026-Task13")
    dir_paths = {name: path / f"task_{name}" for name in ["a", "b", "c"]}
    datasets = {
        name: {
            "training": dir_paths[name] / f"task_{name}_training_set.parquet"
            if name == "b"
            else dir_paths[name] / f"task_{name}_training_set_1.parquet",
            "validation": dir_paths[name] / f"task_{name}_validation_set.parquet",
            "test": dir_paths[name] / f"task_{name}_test_set_sample.parquet",
        }
        for name in ["a", "b", "c"]
    }

    data = pd.read_parquet(datasets["b"]["training"])
    new_data = data.sample(5000)
    new_data["word_count"] = new_data["code"].str.split().str.len()
    new_data["code_len"] = [len(x) for x in parse_data_frame(new_data)]
    del new_data["code"]
    del new_data["generator"]
    del new_data["label"]
    new_data["ratio"] = new_data["word_count"] / new_data["code_len"]
    new_table = new_data.groupby(["language"]).mean()
    new_table["final_ratio"] = new_table["word_count"] / new_table["code_len"]
    print(new_table)
