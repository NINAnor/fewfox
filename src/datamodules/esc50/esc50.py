import glob
import librosa
import torch
import pandas as pd
import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from pytorch_lightning import LightningDataModule

import numpy as np
import librosa
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import os

from easyfsl.samplers import TaskSampler

class AudioDataset(Dataset):
    def __init__(self, root_dir, data_frame, transform=None, target_fs=16000, n_fft=2048, hop_length=512, n_mel=128):
        self.root_dir = root_dir
        self.data_frame = data_frame
        self.transform = transform
        self.target_fs = target_fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mel=n_mel
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data_frame["category"])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Load audio
        audio_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]["filename"])
        label = self.data_frame.iloc[idx]["category"]
        sig, sr = librosa.load(audio_path, sr=self.target_fs, mono=True, res_type="kaiser_fast")

        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=sig, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mel
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Resize spectrogram to fit ResNet50 input
        mel_spec_db = torch.tensor(mel_spec_db, dtype=torch.float32)

        # Duplicate single channel to match ResNet50's 3-channel input
        mel_spec_rgb = mel_spec_db.repeat(3, 1, 1)

        # Encode label as integer
        label = self.label_encoder.transform([label])[0]

        if self.transform:
            mel_spec_rgb = self.transform(mel_spec_rgb)

        return mel_spec_rgb, int(label)
    
    def get_labels(self):
        labels = []

        for i in range(0, len(self.data_frame)):
            label = self.data_frame.iloc[i]["category"]
            label = self.label_encoder.transform([label])[0]
            labels.append(int(label))

        return labels
    
def few_shot_dataloader(
    df, n_way, n_shot, n_query, n_tasks
):
    """
    df: dataset (tensors + labels)
    n_way: number of classes
    n_shot: number of images PER CLASS in the support set
    n_query: number of images PER CLASSS in the query set
    n_tasks: number of episodes (number of times the loader gives the data during a training step)
    """

    sampler = TaskSampler(
        df,
        n_way=n_way,  # number of classes
        n_shot=n_shot,  # Number of images PER CLASS in the support set
        n_query=n_query,  # Number of images PER CLASSS in the query set
        n_tasks=n_tasks,  # Not sure
    )

    loader = DataLoader(
        df,
        batch_sampler=sampler,
        pin_memory=False,
        collate_fn=sampler.episodic_collate_fn,
    )

    return loader

class ESC50DataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        csv_file,
        batch_size,
        split_ratio,
        transform=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.transform = transform
        self.setup()

    def setup(self, stage=None):
        data_frame = pd.read_csv(self.csv_file)
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)
        split_index = int(len(data_frame) * self.split_ratio)
        self.train_set = data_frame.iloc[:split_index, :]
        self.val_set = data_frame.iloc[split_index:, :]

    def train_dataloader(self):
        train_dataset = AudioDataset(
            root_dir=self.root_dir,
            data_frame=self.train_set,
            transform=self.transform,
        )
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_dataset = AudioDataset(
            root_dir=self.root_dir,
            data_frame=self.val_set,
            transform=self.transform,
        )
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    

class ESC50FewShot(LightningDataModule):
    def __init__(
        self,
        root_dir,
        csv_file,
        split_ratio,
        n_way,
        n_shot, 
        n_query, 
        n_task_per_epoch,
        n_validation_tasks,
        transform=None,
        num_workers=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.split_ratio = split_ratio
        self.n_way=n_way
        self.n_shot=n_shot
        self.n_query=n_query
        self.n_task_per_epoch=n_task_per_epoch
        self.n_validation_tasks = n_validation_tasks
        self.transform = transform
        self.num_workers = num_workers
        self._setup()

    def _setup(self, stage=None):
        data_frame = pd.read_csv(self.csv_file)
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)
        split_index = int(len(data_frame) * self.split_ratio)
        self.train_set = data_frame.iloc[:split_index, :]
        self.val_set = data_frame.iloc[split_index:, :]

    def train_dataloader(self):
        train_dataset = AudioDataset(
            root_dir=self.root_dir,
            data_frame=self.train_set,
            transform=self.transform,
        )

        train_sampler = TaskSampler(train_dataset, 
                                    self.n_way, 
                                    self.n_shot, 
                                    self.n_query, 
                                    self.n_task_per_epoch)
        
        return DataLoader(train_dataset,
            batch_sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=train_sampler.episodic_collate_fn,
        )

    def val_dataloader(self):
        val_dataset = AudioDataset(
            root_dir=self.root_dir,
            data_frame=self.val_set,
            transform=self.transform,
        )

        val_sampler = TaskSampler(val_dataset, 
                                  self.n_way, 
                                  self.n_shot, 
                                  self.n_query, 
                                  self.n_validation_tasks)

        return DataLoader(val_dataset, 
                          batch_sampler=val_sampler, 
                          num_workers=self.num_workers, 
                          pin_memory=True, 
                          collate_fn=val_sampler.episodic_collate_fn,)

def few_shot_dataloader(root_dir, data_frame, n_way, n_shot, n_query, n_tasks, transform = None): 
    """
    root_dir: directory where the audio data is stored
    data_frame: path to the label file
    n_way: number of classes
    n_shot: number of images PER CLASS in the support set
    n_query: number of images PER CLASSS in the query set
    n_tasks: number of episodes (number of times the loader gives the data during a training step)
    """       
    
    df = AudioDataset(
        root_dir=root_dir, data_frame=data_frame, transform=transform
    )

    sampler = TaskSampler(
        df, 
        n_way=n_way, # number of classes
        n_shot=n_shot, # Number of images PER CLASS in the support set
        n_query=n_query, # Number of images PER CLASSS in the query set
        n_tasks=n_tasks # Not sure
    )

    loader = DataLoader(
        df,
        batch_sampler=sampler,
        pin_memory=False,
        collate_fn=sampler.episodic_collate_fn
    )

    return loader


class miniECS50DataModule(LightningDataModule):
    def __init__(
        self,
        root_dir_train: str = "/data/ESC50mini/audio/train",
        root_dir_val: str = "/data/ESC50mini/audio/val",
        root_dir_test: str = "/data/ESC50mini/audio/test",
        csv_file_train: str = "/data/ESC50mini/meta/esc50mini_train.csv",
        csv_file_val: str = "/data/ESC50mini/meta/esc50mini_val.csv",
        csv_file_test: str = "/data/ESC50mini/meta/esc50mini_test.csv",
        n_task_train: int = 100,
        n_task_val: int = 100,
        n_task_test: int = 10 ,
        transform=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.root_dir_train = root_dir_train
        self.root_dir_val = root_dir_val
        self.root_dir_test = root_dir_test
        self.csv_file_train = csv_file_train
        self.csv_file_val = csv_file_val
        self.csv_file_test = csv_file_test
        self.n_task_train = n_task_train
        self.n_task_val = n_task_val
        self.n_task_test = n_task_test
        self.transform = transform

        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_set= pd.read_csv(self.csv_file_train)
        self.val_set = pd.read_csv(self.csv_file_val)
        self.test_set = pd.read_csv(self.csv_file_test)

    def train_dataloader(self):

        train_loader = few_shot_dataloader(self.root_dir_train, 
                                           self.train_set, 
                                           n_way=5, 
                                           n_shot=5, 
                                           n_query=5, 
                                           n_tasks=self.n_task_train, 
                                           transform=self.transform)
        return train_loader

    def val_dataloader(self):

        val_loader = few_shot_dataloader(self.root_dir_val, 
                                           self.val_set, 
                                           n_way=5, 
                                           n_shot=3, 
                                           n_query=2, 
                                           n_tasks=self.n_task_val, 
                                           transform=self.transform)
        return val_loader
    
    def test_dataloader(self):

        test_loader = few_shot_dataloader(self.root_dir_test, 
                                           self.test_set, 
                                           n_way=5, 
                                           n_shot=5, 
                                           n_query=20, 
                                           n_tasks=self.n_task_test, 
                                           transform=self.transform)
        return test_loader