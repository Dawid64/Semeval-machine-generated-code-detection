from pathlib import Path
from typing import Literal
import pandas as pd
import numpy as np
from src.data_processing import load_data, parse_data_frame
from torch import nn
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from tqdm.auto import trange
from torchmetrics.classification import Accuracy, AUROC


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        num_epochs: int = 50,
        batch_size: int = 16,
        dataset_name: Literal["a", "b", "c"] = "a",
        dataset_part: int | None = 10_000,
        save_path: str | Path = "model.pth",
        num_classes: int = 2,
        early_stopping_patience: int | None = None,
        weight_classes: bool = False,
    ) -> None:
        torch.set_float32_matmul_precision = "medium"
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dataset_part = dataset_part
        self.num_classes = num_classes
        self.early_stopping_patience = early_stopping_patience
        self.training_dataset = self.prepare_dataset(load_data(dataset_name, "train"))
        self.validation_dataset = self.prepare_dataset(load_data(dataset_name, "val"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        if weight_classes:
            label_counts = load_data(dataset_name, "train").label.value_counts().sort_index().to_numpy()
            weights = label_counts.sum() / (label_counts.__len__() * label_counts)
            weights = torch.tensor(weights).to(self.device, dtype=torch.float)
        else:
            weights = None
        self.criterion = torch.nn.CrossEntropyLoss(weight=weights) if self.num_classes > 2 else torch.nn.BCELoss(weight=weights)

        self.save_path = save_path

        self.train_loader = self.create_loader(self.training_dataset, True)
        self.val_loader = self.create_loader(self.validation_dataset, False)

        self.acc = Accuracy(task="multiclass" if self.num_classes > 2 else "binary", num_classes=self.num_classes).to(self.device)
        self.auc = AUROC(task="multiclass" if self.num_classes > 2 else "binary", num_classes=self.num_classes).to(self.device)

    def create_loader(self, dataset, shuffle):
        graphs = []
        for _, row in dataset.iterrows():
            g = row.code_tree
            g.y = torch.tensor([row.label], dtype=torch.long)
            graphs.append(g)
        return DataLoader(graphs, batch_size=self.batch_size, shuffle=shuffle)

    def prepare_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if self.dataset_part is not None:
            dataset = dataset.sample(self.dataset_part)
        dataset["code_tree"] = parse_data_frame(dataset)
        dataset.drop(["code", "language"], axis=1)
        return dataset

    def train(self):
        iterator = trange(self.num_epochs, desc="epoch 0 loss: <infinite> acc: 0%")
        lowest_loss = np.inf
        non_improvement_counter = 0
        for epoch in iterator:
            train_loss = self.train_loop(epoch)
            val_loss = self.val_loop()
            if val_loss < lowest_loss:
                lowest_loss = val_loss
                non_improvement_counter = 0
            else:
                non_improvement_counter += 1
                if self.early_stopping_patience is not None and non_improvement_counter >= self.early_stopping_patience:
                    print(f"Training early stopped. Lowest validation loss was {lowest_loss}")
                    break

            acc = self.acc.compute()
            auc = self.auc.compute()
            iterator.set_description(
                f"epoch {epoch + 1} train loss: {train_loss:.4g} val loss: {val_loss:.4g} acc: {acc * 100:.4g}% auc: {auc:.4g}"
            )
            self.acc.reset()
            self.auc.reset()
            torch.save(self.model.state_dict(), self.save_path)

    def train_loop(self, epoch):
        losses = []
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            if epoch == 0 and i == 0:
                print(summary(self.model, batch.to(self.device)))
            self.optimizer.zero_grad()
            out = self.model(batch.to(self.device))
            if self.num_classes == 2:
                out = out.squeeze()
            target = batch.y if self.num_classes > 2 else batch.y.float()
            loss = self.criterion(out, target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return sum(losses) / len(losses)

    def val_loop(self):
        losses = []
        self.model.eval()
        with torch.inference_mode():
            for batch in self.val_loader:
                out = self.model(batch.to(self.device))
                if self.num_classes == 2:
                    out = out.squeeze()
                target = batch.y if self.num_classes > 2 else batch.y.float()
                loss = self.criterion(out, target)
                self.acc(out, batch.y)
                self.auc(out, batch.y)
                losses.append(loss.item())
        return sum(losses) / len(losses)


if __name__ == "__main__":
    from src.models.graph_v1 import GraphClassifier

    trainer = Trainer(GraphClassifier(5, 2))
    trainer.train()
