import shutil
import kagglehub
from pathlib import Path


def download():
    if Path("data", "SemEval-2026-Task13").exists():
        return
    path = kagglehub.dataset_download("daniilor/semeval-2026-task13")
    print("Path to dataset files:", path)

    data = Path("data")
    data.mkdir(exist_ok=True)
    shutil.move(Path(path, "SemEval-2026-Task13"), data)
    print(f"Moved files to {data}")


if __name__ == "__main__":
    download()
