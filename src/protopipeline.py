from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pathlib
import os

import torch.nn as nn

from datamodules.esc50.mini_esc50 import miniESC50DataModule
from datamodules.esc50.embeddings_esc50 import EmbeddingsMiniESC50DataModule
from torchvision.models import resnet50, ResNet50_Weights
from models.protonet import ProtoNetworkModel
import hydra

from bacpipe.main import get_embeddings

class Identity(nn.Module):
    def forward(self, x):
        return x
    
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):

    root_dir = pathlib.Path(cfg.paths.MINI_ESC50_PATH)

    if cfg.train.DATALOADER == "classic":

        data_module = miniESC50DataModule(
            root_dir_train = root_dir / "audio/train",
            root_dir_val = root_dir / "audio/val",
            root_dir_test = root_dir / "audio/test",
            csv_file_train = root_dir / "meta/esc50mini_train.csv",
            csv_file_val = root_dir / "meta/esc50mini_val.csv",
            csv_file_test = root_dir / "meta/esc50mini_test.csv",
            n_task_train = 50,
            n_task_val = 10,
            n_task_test = 10
        )

    elif cfg.train.DATALOADER == "bacpipe":

        # Generate the embeddings (they will be saved in a folder)
        print("[INFO] GENERATING THE TRAINING EMBEDDINGS")
        loader, _ = get_embeddings("birdnet", root_dir / "audio/train", check_if_primary_combination_exists=True)
        train_embed_dir = loader.embed_dir

        print("[INFO] GENERATING THE VALIDATION EMBEDDINGS")
        loader, _ = get_embeddings("birdnet", root_dir / "audio/val", check_if_primary_combination_exists=True)
        val_embed_dir = loader.embed_dir

        print("[INFO] GENERATING THE TEST EMBEDDINGS")
        loader, _ = get_embeddings("birdnet", root_dir / "audio/test", check_if_primary_combination_exists=True)
        test_embed_dir = loader.embed_dir

        print("[INFO] CREATING THE DATALOADER")
        data_module = EmbeddingsMiniESC50DataModule(
            embed_dir_train = str(train_embed_dir),
            embed_dir_val   = str(val_embed_dir),
            embed_dir_test  = str(test_embed_dir),
            csv_file_train  = str(root_dir / "meta/esc50mini_train.csv"),
            csv_file_val    = str(root_dir / "meta/esc50mini_val.csv"),
            csv_file_test   = str(root_dir / "meta/esc50mini_test.csv"),
            n_task_train    = 50,
            n_task_val      = 10,
            n_task_test     = 10
        )

    ######################################
    # TRAIN USING THE PROTOTYPICAL MODEL #
    ######################################
        
    # Define ResNet50 as feature extractor
    print("[INFO] TRAINING THE MODEL")
    if cfg.train.DATALOADER == "classic":
        # Use ResNet50 for spectrogram images.
        from torchvision.models import resnet50, ResNet50_Weights
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        ft_entire_network = True
    else:
        # Use an identity module to pass through the precomputed embeddings.
        backbone = Identity()
        ft_entire_network = False

    protonetwork = ProtoNetworkModel(backbone_model=backbone, 
                                     n_way=5, 
                                     ft_entire_network=ft_entire_network)
    
    trainer = pl.Trainer(max_epochs=10,
                         default_root_dir=os.getcwd())
    trainer.fit(protonetwork, datamodule=data_module)


if __name__ == "__main__":
    main()
