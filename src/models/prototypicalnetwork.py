import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy, F1Score

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from easyfsl.methods import PrototypicalNetworks

class ProtoNetworkModel(pl.LightningModule):
    def __init__(
        self,
        backbone_model,
        n_way: int = 20,
        milestones: int = 5,
        lr: float = 1e-5,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        ft_entire_network: bool = True,
        **kwargs,
    ) -> None:
        """TransferLearningModel.
        Args:
            lr: Initial learning rate
        """
        super().__init__()
        self.backbone_model = backbone_model
        self.n_way = n_way
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        self.milestones = milestones
        self.ft_entire_network = ft_entire_network

        self._build_model()
        self.save_hyperparameters()

        self.train_acc = Accuracy(task="multiclass", num_classes=self.n_way)
        self.valid_acc = Accuracy(task="multiclass", num_classes=self.n_way)
        self.train_f1 = F1Score(task="multiclass", num_classes=self.n_way)
        self.valid_f1 = F1Score(task="multiclass", num_classes=self.n_way)


    def _build_model(self):
        self.model = PrototypicalNetworks(self.backbone_model)
        return self.model
    
    def forward(
            self, 
            support_images, # shape is n_way
            support_labels,
            query_images):
        
        self.model.process_support_set(
            support_images, support_labels)
        print(self.model.prototypes.shape)
        classification_scores = self.model(query_images) # shape should be [n_way, n_query]

        return classification_scores

    def loss(self, lprobs, labels):
        self.loss_func = nn.CrossEntropyLoss()
        return self.loss_func(lprobs, labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        support_images, support_labels, query_images, query_labels, _ = batch
        print(support_images.shape)
        classification_scores = self.forward(
            support_images, support_labels, query_images
        )

        # 2. Compute loss
        train_loss = self.loss(classification_scores.requires_grad_(True), query_labels)
        self.log("train_loss", train_loss, prog_bar=True)

        # 3. Compute accuracy:
        predicted_labels = torch.max(classification_scores, 1)[1]
        self.log(
            "train_acc", self.train_acc(predicted_labels, query_labels), prog_bar=True
        )
        self.log(
            "train_f1", self.train_f1(predicted_labels, query_labels), prog_bar=True
        )

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        support_images, support_labels, query_images, query_labels, _ = batch
        classification_scores = self.forward(
            support_images, support_labels, query_images
        )

        # 2. Compute loss
        self.log(
            "val_loss", self.loss(classification_scores, query_labels), prog_bar=True
        )

        # 3. Compute accuracy:
        predicted_labels = torch.max(classification_scores, 1)[1]
        self.log(
            "val_acc", self.valid_acc(predicted_labels, query_labels), prog_bar=True
        )
        self.log(
            "valid_f1", self.valid_f1(predicted_labels, query_labels), prog_bar=True
        )

    def configure_optimizers(self):
        if self.ft_entire_network:
            optimizer = optim.AdamW(
                self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01
            )

        else:
            optimizer = optim.AdamW(
                self.fc.parameters,
                lr=self.lr,
                betas=(0.9, 0.98),
                weight_decay=0.01,
            )
        return optimizer