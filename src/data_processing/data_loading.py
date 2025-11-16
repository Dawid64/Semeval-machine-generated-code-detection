import shutil
from typing import Literal
import kagglehub
from pathlib import Path

import pandas as pd


def download():
    if Path("data", "SemEval-2026-Task13").exists():
        return
    path = kagglehub.dataset_download("daniilor/semeval-2026-task13")
    print("Path to dataset files:", path)

    data = Path("data")
    data.mkdir(exist_ok=True)
    shutil.move(Path(path, "SemEval-2026-Task13"), data)
    print(f"Moved files to {data}")


def load_data(
    task: Literal["a", "b", "c"], data_part: Literal["train", "val", "test"]
) -> pd.DataFrame:
    if Path("data").exists():
        path = Path("data", "SemEval-2026-Task13")
    elif Path("..", "data").exists():
        path = Path("..", "data", "SemEval-2026-Task13")
    else:
        raise FileNotFoundError(
            "Dataset not found, to download use `python -m src.data_processing.data_loading`"
        )
    dir_paths = {name: path / f"task_{name}" for name in ["a", "b", "c"]}
    datasets = {
        name: {
            "train": dir_paths[name] / f"task_{name}_training_set.parquet"
            if name == "b"
            else dir_paths[name] / f"task_{name}_training_set_1.parquet",
            "val": dir_paths[name] / f"task_{name}_validation_set.parquet",
            "test": dir_paths[name] / f"task_{name}_test_set_sample.parquet",
        }
        for name in ["a", "b", "c"]
    }
    data = pd.read_parquet(datasets[task][data_part])
    data.drop(["generator"], axis=1, inplace=True)
    return data


if __name__ == "__main__":
    download()
