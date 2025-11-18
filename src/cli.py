from typing import Literal
import click
import torch
from src.data_processing import download
from src.data_processing.code_parser import SUPPORTED_LANGUAGES, create_graph
from src.train import Trainer
from src.models import __all_models__, AVAILABLE_MODELS


@click.group()
def cli():
    pass


@cli.add_command
@click.command()
@click.option("--dataset", "-d", default="a", help="Task name <'a', 'b', 'c'>")
@click.option(
    "--dataset_size",
    default=None,
    type=int,
    help="Number of samples taken for training",
)
@click.option("--save", default="model.pth", help="Path to where to save the model")
@click.argument("name")
def train(
    dataset: Literal["a", "b", "c"],
    dataset_size: None | int,
    save: str,
    name: AVAILABLE_MODELS,
):
    model = __all_models__[name]
    trainer = Trainer(
        model, dataset_name=dataset, dataset_part=dataset_size, save_path=save
    )
    trainer.train()


@cli.add_command
@click.command()
@click.option(
    "--from_file", "-f", default="model.pth", help="Path to where the model is saved"
)
@click.option("--language", "-l", default="Python", help="Uppercase code language name")
@click.argument("name")
@click.argument("path")
def check(
    from_file: str,
    language: SUPPORTED_LANGUAGES,
    name: AVAILABLE_MODELS,
    path: str,
):
    with open(path, "r") as file:
        code = "\n".join(file.readlines())
    with open(from_file, "rb") as file:
        state_dict = torch.load(file, "cpu")
    code_tree = create_graph(code, language)
    model = __all_models__[name]
    model.load_state_dict(state_dict)
    output = model(code_tree)
    print(output)


cli.add_command(click.command(download))

if __name__ == "__main__":
    cli()
