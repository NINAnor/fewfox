#!/usr/bin/env python3

import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from sklearn.manifold import TSNE

from datamodules.esc50.mini_esc50 import miniECS50DataModule
from easyfsl.methods import PrototypicalNetworks
from torchvision.models import resnet50, ResNet50_Weights
import torch

import hydra

def evaluate_on_one_task(
    model,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
):
    """
    Returns the number of correct predictions of query labels, the total 
    number of predictions, and the coordinates of the prototypes and query images.
    """

    # Get the prototypes
    model.compute_prototypes_and_store_support_set(support_images.cuda(), support_labels.cuda())
    prototypes = model.prototypes

    query_embeddings = model.compute_features(query_images.cuda())
    query_embeddings = query_embeddings.detach().cpu()
    prototypes = prototypes.detach().cpu()
    return (
        torch.max(
            model(
                query_images.cuda(),
            )
            .detach()
            .data,
            1,
        )[1]
        == query_labels.cuda()
    ).sum().item(), len(query_labels), prototypes, query_embeddings


def evaluate(model, data_loader: DataLoader):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0
    all_prototypes = []
    all_query_embeddings = []
    all_query_labels = []

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    model.eval().to("cuda")
    with torch.no_grad():
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):

            correct, total, prototypes, query_embeddings = evaluate_on_one_task(
                model, support_images, support_labels, query_images, query_labels
            )

            total_predictions += total
            correct_predictions += correct
            all_prototypes.append(prototypes)
            all_query_embeddings.append(query_embeddings)
            all_query_labels.append(query_labels)

    all_prototypes = torch.cat(all_prototypes, dim=0)
    all_query_embeddings = torch.cat(all_query_embeddings, dim=0)
    all_query_labels = torch.cat(all_query_labels, dim=0)

    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
    )

    return all_prototypes, all_query_embeddings, all_query_labels

def get_2d_features(features, perplexity):
    return TSNE(n_components=2, perplexity=perplexity).fit_transform(features)

def get_figure(features_2d, labels, fig_name):

    query_2d = features_2d[5:]
    query_labels = labels[5:]

    proto_2d = features_2d[:5]
    proto_labels = labels[:5]

    fig = sns.scatterplot(x=query_2d[:, 0], y=query_2d[:, 1], hue=query_labels, palette="deep")
    sns.scatterplot(x=proto_2d[:, 0], y=proto_2d[:, 1], hue=proto_labels, palette="deep", marker='s', s=100)
    
    sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
    fig.get_figure().savefig(fig_name, bbox_inches="tight")
    plt.show()

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):

    model = PrototypicalNetworks(backbone=resnet50(weights=ResNet50_Weights.IMAGENET1K_V1))
    ckpt = torch.load(config.paths.MODEL_PATH)
    state_dict = ckpt["state_dict"]

    # Create a new state dict with renamed keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # Replace "backbone_model." with "backbone." if present.
        if key.startswith("backbone_model."):
            new_key = key.replace("backbone_model.", "backbone.")
        # If the keys are prefixed with "model.backbone.", replace them too.
        elif key.startswith("model.backbone."):
            new_key = key.replace("model.backbone.", "backbone.")
        new_state_dict[new_key] = value

    # Load state dic
    model.load_state_dict(new_state_dict)

    test_loader = miniECS50DataModule(n_task_test=1)
    test_loader = test_loader.test_dataloader()

    all_prototypes, all_query_embeddings, all_query_labels = evaluate(model, test_loader)

    # Select the first batch of embeddings - IN WORK MEAN ACROSS EACH TENSOR
    prototype_s = all_prototypes[:5,:]
    query_embeddings_s = all_query_embeddings[:100,:]
    query_labels_s = all_query_labels_r = all_query_labels[0:100]

    # For TSNE, concatenate prototypes and then queries
    proto_query = torch.cat([prototype_s, query_embeddings_s])
    all_labels = torch.cat([torch.tensor([5,5,5,5,5,]), query_labels_s])

    features_2d = get_2d_features(proto_query, perplexity=5)
    print(features_2d.shape)
    get_figure(features_2d, all_labels, "protoembeddings.png")

if __name__ == "__main__":

    main()