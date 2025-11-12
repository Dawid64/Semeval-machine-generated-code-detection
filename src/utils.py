import shutil
import kagglehub
from pathlib import Path


def download():
    #
    if Path("SemEval-2026-Task13").exists():
        return
    path = kagglehub.dataset_download("daniilor/semeval-2026-task13")

    print("Path to dataset files:", path)

    path = f"{Path.home()}/.cache/kagglehub/datasets/daniilor/semeval-2026-task13/versions/3"

    shutil.move(Path(path, "SemEval-2026-Task13"), Path.cwd() / "data")


if __name__ == "__main__":
    download()
