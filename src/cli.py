import click
from src.data_processing import download


@click.group()
def cli():
    pass


@click.command()
@click.argument("name")
def train(name: str):
    click.echo("Hello World")


cli.add_command(train)
cli.add_command(click.command(download))

if __name__ == "__main__":
    cli()
