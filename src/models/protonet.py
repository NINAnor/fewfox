import torch
from torch import nn, optim
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
from easyfsl.methods import PrototypicalNetworks

class ProtoNetworkModel(pl.LightningModule):
    def __init__(
        self,
        backbone_model,
        n_way: int = 20,
        embedding_dim: int = 128,      # dimension of your precomputed embeddings
        projection_dim: int = 512,       # dimension expected by your prototypical network
        milestones: int = 5,
        lr: float = 1e-5,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        ft_entire_network: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            backbone_model: The backbone model (or Identity) for feature extraction.
            n_way: Number of classes in each task.
            embedding_dim: Dimensionality of the input embeddings.
            projection_dim: Dimensionality to project the embeddings to.
            ft_entire_network: Whether to fine-tune the entire network.
        """
        super().__init__()
        self.backbone_model = backbone_model
        self.n_way = n_way
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        self.milestones = milestones
        self.ft_entire_network = ft_entire_network
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim

        # Build the prototypical network with the provided backbone.
        self.model = PrototypicalNetworks(self.backbone_model)
        
        self.save_hyperparameters(ignore=['backbone_model'])
        
        self.train_acc = Accuracy(task="multiclass", num_classes=self.n_way)
        self.valid_acc = Accuracy(task="multiclass", num_classes=self.n_way)
        self.train_f1 = F1Score(task="multiclass", num_classes=self.n_way)
        self.valid_f1 = F1Score(task="multiclass", num_classes=self.n_way)

    def forward(self, support_images, support_labels, query_images):
        self.model.process_support_set(support_images, support_labels)
        classification_scores = self.model(query_images)
        return classification_scores

    def loss(self, lprobs, labels):
        loss_func = nn.CrossEntropyLoss()
        return loss_func(lprobs, labels)

    def training_step(self, batch, batch_idx):
        support_images, support_labels, query_images, query_labels, _ = batch
        classification_scores = self.forward(support_images, support_labels, query_images)
        train_loss = self.loss(classification_scores.requires_grad_(True), query_labels)
        self.log("train_loss", train_loss, prog_bar=True)
        predicted_labels = torch.max(classification_scores, 1)[1]
        self.log("train_acc", self.train_acc(predicted_labels, query_labels), prog_bar=True)
        self.log("train_f1", self.train_f1(predicted_labels, query_labels), prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        support_images, support_labels, query_images, query_labels, _ = batch
        classification_scores = self.forward(support_images, support_labels, query_images)
        self.log("val_loss", self.loss(classification_scores, query_labels), prog_bar=True)
        predicted_labels = torch.max(classification_scores, 1)[1]
        self.log("val_acc", self.valid_acc(predicted_labels, query_labels), prog_bar=True)
        self.log("valid_f1", self.valid_f1(predicted_labels, query_labels), prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            betas=(0.9, 0.98), 
            weight_decay=0.01
        )
        return optimizer
