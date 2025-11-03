import os
import shutil
import kagglehub


def download():
    path = kagglehub.dataset_download("daniilor/semeval-2026-task13")

    print("Path to dataset files:", path)

    path = (
        "/Users/dawid/.cache/kagglehub/datasets/daniilor/semeval-2026-task13/versions/3"
    )

    shutil.move(os.path.join(path, "SemEval-2026-Task13"), os.getcwd())


if __name__ == "__main__":
    download()
