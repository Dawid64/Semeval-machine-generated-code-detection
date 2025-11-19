# SemEval-2026 Machine Generated Code Detection

## Installation

To clone repository and prepare environment with uv do the following:

```sh
git clone --recurse-submodules https://github.com/Dawid64/Semeval-machine-generated-code-detection.git
cd Semeval-machine-generated-code-detection

uv sync
uv run semeval download
```

## Usage

There is a CLI tool that allows to use most important functionalities of the project:

```sh
uv run semeval --help
```

Currently cli can be used to:

- train - Train the model
- check - Use model to classify given file

Example:

```sh
source .venv/bin/activate
semeval train --dataset c --dataset_size 200 graph_v1
```