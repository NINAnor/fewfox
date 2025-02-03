from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import torch.nn as nn

from datamodules.esc50 import ESC50FewShot
from torchvision.models import resnet50, ResNet50_Weights
from models.prototypicalnetwork import ProtoNetworkModel
import hydra

from easyfsl.methods import PrototypicalNetworks, FewShotClassifier

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):

    n_way=5
    n_shot=3
    n_query=2
    n_task_per_epoch=100
    n_validation_tasks=100
    # REMEMBER FOR RESNET
    #mel_spec_rgb = mel_spec_resized.repeat(3, 1, 1)

    # Each batch returns:
    #    tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
    #        - support images of shape (n_way * n_shot, n_channels, height, width),
    #        - their labels of shape (n_way * n_shot),
    #        - query images of shape (n_way * n_query, n_channels, height, width)
    #        - their labels of shape (n_way * n_query),
    #        - the dataset class ids of the class sampled in the episode
    data = ESC50FewShot(       
        root_dir=cfg.paths.ROOT_DIR_ESC50,
        csv_file=cfg.paths.CSV_FILE_ESC50,
        split_ratio=0.6,
        n_way=n_way,
        n_shot=n_shot, 
        n_query=n_query, 
        n_task_per_epoch=n_task_per_epoch,
        n_validation_tasks=n_validation_tasks
        )
        
    train_data = data.train_dataloader()
    val_data = data.val_dataloader()

    # Define ResNet50 as feature extractor
    resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    protonetwork = ProtoNetworkModel(backbone_model=resnet_model, 
                                     n_way=n_way, 
                                     ft_entire_network=True)
    
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(protonetwork, train_dataloaders=train_data, val_dataloaders=val_data, )


    protonet = PrototypicalNetworks(resnet_model)


if __name__ == "__main__":
    main()
