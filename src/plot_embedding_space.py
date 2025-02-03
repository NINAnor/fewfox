import hydra
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from sklearn.manifold import TSNE
import plotly.express as px

from models.ft_vanilla import TransferLearningModel
from datamodules.esc50 import ESC50DataModule
from torchvision.models import resnet50

torch.backends.cudnn.benchmark = True

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg):
    # Get the dataset
    esc50 = ESC50DataModule(
        root_dir=cfg.paths.ROOT_DIR_ESC50,
        csv_file=cfg.paths.CSV_FILE_ESC50,
        batch_size=cfg.train.BATCH_SIZE,
        split_ratio=cfg.train.SPLIT_RATIO,
    )

    train_set = esc50.train_dataloader()

    # Define ResNet50 as feature extractor
    resnet_model = resnet50(pretrained=True)
    resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])  # Remove classification layer

    # Initialize transfer learning model
    model = TransferLearningModel(
        feature_extractor=resnet_model,
        num_target_classes=50,
        embedding_dim=2048,  # ResNet50 output before the classification layer
    )

    # Collect embeddings and labels
    all_embeddings = []
    all_labels = []

    for batch in train_set:
        tensor, labels = batch
        embeddings = model.extract_embeddings(tensor).squeeze(-1).squeeze(-1)  # Flatten embeddings
        all_embeddings.append(embeddings)  # Keep embeddings as tensors
        all_labels.append(labels)

    # Concatenate all embeddings and labels (as tensors)
    all_embeddings = torch.cat(all_embeddings)  # Concatenate tensors along the batch dimension
    all_labels = torch.cat(all_labels)

    # Convert to numpy for dimensionality reduction
    all_embeddings_np = all_embeddings.detach().cpu().numpy()
    all_labels_np = all_labels.detach().cpu().numpy()

    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=7)
    embeddings_2d = tsne.fit_transform(all_embeddings_np)

    # Plot using Plotly
    fig = px.scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        color=all_labels_np,
        labels={'color': 'Class'},
        title="Embedding Space Visualization (t-SNE)",
    )
    fig.update_layout(
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        template="plotly"
    )
    fig.show()

if __name__ == "__main__":
    train()
