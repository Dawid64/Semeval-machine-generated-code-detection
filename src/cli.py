from typing import Literal
import fire
import torch
from src.data_processing import download
from src.data_processing.code_parser import SUPPORTED_LANGUAGES, create_graph
from src.train import Trainer
from src.models import __all_models__, AVAILABLE_MODELS


def train(
    name: AVAILABLE_MODELS,
    save: str = "model.pth",
    dataset: Literal["a", "b", "c"] = "a",
    dataset_size: None | int = None,
):
    """
    Trains the model
    Args:
        name (AVAILABLE_MODELS): Name of the model
        save (str, optional): where to save the model. Defaults to "model.pth".
        dataset (Literal[&quot;a&quot;, &quot;b&quot;, &quot;c&quot;], optional): name of dataset. Defaults to "a".
        dataset_size (None | int, optional): Number of samples that will be taken into dataset, if None it takes whole dataset. Defaults to None.
    """
    model = __all_models__[name]
    trainer = Trainer(
        model, dataset_name=dataset, dataset_part=dataset_size, save_path=save
    )
    trainer.train()


def check(
    name: AVAILABLE_MODELS,
    language: SUPPORTED_LANGUAGES,
    path: str,
    model_path: str = "model.pth",
):
    """Uses given from user model to perform interference on given file

    Args:
        name (AVAILABLE_MODELS): Name of model.
        language (SUPPORTED_LANGUAGES): Language in uppercase for example Python.
        path (str): path to the file
        model_path (str, optional): Path to the model. Defaults to "model.pth".
    """
    with open(path, "r") as file:
        code = "\n".join(file.readlines())
    with open(model_path, "rb") as file:
        state_dict = torch.load(file, "cpu")
    code_tree = create_graph(code, language)
    model = __all_models__[name]
    model.load_state_dict(state_dict)
    output = model(code_tree)
    print(output)


def cli():
    fire.Fire(
        {
            "download": download,
            "train": train,
            "check": check,
        }
    )


if __name__ == "__main__":
    cli()
