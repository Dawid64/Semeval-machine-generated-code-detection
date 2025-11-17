from pathlib import Path
from typing import Literal
import pandas as pd
from src.data_processing import load_data, parse_data_frame
from torch import nn
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from tqdm.auto import trange


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        num_epochs:int= 50,
        batch_size:int = 16,
        dataset_name: Literal["a", "b", "c"] = "a",
        dataset_part: int | None = 10_000,
        save_path: str | Path = "model.pth",
    ) -> None:
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dataset_part = dataset_part
        self.training_dataset = self.prepare_dataset(load_data(dataset_name, "train"))
        self.validation_dataset = self.prepare_dataset(load_data(dataset_name, "val"))
        self.device = "cpu" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.save_path = save_path

    def prepare_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if self.dataset_part is not None:
            dataset = dataset.sample(self.dataset_part)
        dataset["code_tree"] = parse_data_frame(dataset)
        dataset.drop(["code", "language"], axis=1)
        return dataset

    def get_acc(self, num_samples: int = 1000):
        hits = 0
        for _, row in self.validation_dataset.sample(num_samples).iterrows():
            pred = torch.argmax(self.model(row.code_tree.to(self.device)))
            if pred == row.label:
                hits += 1
        return hits / num_samples

    def train(self):
        graphs = []
        for _, row in self.training_dataset.iterrows():
            g = row.code_tree
            g.y = torch.tensor([row.label], dtype=torch.long)
            graphs.append(g)

        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=True)

        iterator = trange(self.num_epochs, desc="epoch 0 loss: <infinite> acc: 0%")
        for epoch in iterator:
            total_loss = 0
            self.model.train()

            for i, batch in enumerate(loader):
                if epoch == 0 and i == 0:
                    print(summary(self.model, batch.to(self.device)))
                self.optimizer.zero_grad()
                out = self.model(batch.to(self.device))
                loss = self.criterion(out, batch.y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            iterator.set_description(
                f"epoch {epoch + 1} loss: {total_loss:.1f} acc: {self.get_acc() * 100:.1f}%"
            )
            torch.save(self.model.state_dict(), self.save_path)


if __name__ == "__main__":
    from src.models.graph_v1 import GraphClassifier

    trainer = Trainer(GraphClassifier(5, 2))
    trainer.train()
