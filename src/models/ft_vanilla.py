import torch
from torch import nn, optim
from torchmetrics import Accuracy

import pytorch_lightning as pl

class TransferLearningModel(pl.LightningModule):
    def __init__(
        self,
        feature_extractor: nn.Module,
        num_target_classes: int = 50,
        milestones: int = 5,
        batch_size: int = 32,
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        ft_entire_network: bool = False,  # should all the layers be trained?
        embedding_dim: int = 512,  
        **kwargs,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.milestones = milestones
        self.num_target_classes = num_target_classes
        self.ft_entire_network = ft_entire_network
        self.embedding_dim = embedding_dim

        # Assign the feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()  

        # Classifier head
        self.fc = nn.Linear(self.embedding_dim, self.num_target_classes)

        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_target_classes)
        self.valid_acc = Accuracy(task="multiclass", num_classes=self.num_target_classes)
        self.save_hyperparameters()

    def extract_embeddings(self, x):
        """Extract embeddings using the feature extractor."""
        with torch.no_grad():
            embeddings = self.feature_extractor(x)
        return embeddings

    def forward(self, x):
        """Forward pass."""
        x = self.extract_embeddings(x)
        x = self.fc(x)
        return x

    def loss(self, lprobs, labels):
        self.loss_func = nn.CrossEntropyLoss()
        return self.loss_func(lprobs, labels)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_probs = self.forward(x)

        train_loss = self.loss(y_probs, y_true)
        self.log("train_acc", self.train_acc(y_probs, y_true), prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_probs = self.forward(x)

        self.log("val_loss", self.loss(y_probs, y_true), prog_bar=True)
        self.log("val_acc", self.valid_acc(y_probs, y_true), prog_bar=True)

    def configure_optimizers(self):
        if self.ft_entire_network:
            optimizer = optim.AdamW(
                [
                    {"params": self.feature_extractor.parameters()},
                    {"params": self.fc.parameters()},
                ],
                lr=self.lr,
                betas=(0.9, 0.98),
                weight_decay=0.01,
            )
        else:
            optimizer = optim.AdamW(
                self.fc.parameters(), lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01
            )

        return optimizer


