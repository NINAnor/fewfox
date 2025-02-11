import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from easyfsl.samplers import TaskSampler
from pytorch_lightning import LightningDataModule
import pandas as pd
import numpy as np
import os

class EmbeddingFileDataset(Dataset):
    def __init__(self, embed_dir, data_frame, transform=None):
        """
        Args:
            embed_dir (str or Path): The directory where embedding files are stored.
            data_frame (str or pd.DataFrame): CSV file path or DataFrame with at least 'filename' and 'category' columns.
            transform: Optional transform to apply to each embedding.
        """
        # Load CSV if a file path is provided.
        if isinstance(data_frame, str):
            self.data_frame = pd.read_csv(data_frame)
        else:
            self.data_frame = data_frame.reset_index(drop=True)
            
        self.embed_dir = os.path.abspath(embed_dir)
        self.transform = transform

        # Fit the label encoder on the 'category' column.
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data_frame["category"])

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        filename = row["filename"]  # e.g. "file.wav" or "file"
        label = row["category"]

        # Remove the original extension (if any) and append '_birdnet.npy'
        base_name = os.path.splitext(filename)[0]
        embed_file = os.path.join(self.embed_dir, base_name + "_birdnet.npy")
        
        # Load the embedding from file (assuming it was saved as a NumPy array).
        embedding = np.load(embed_file)

        # ECS50 is composed of recordings of 6 seconds. Using BirdNET, bacpipe will create arrays of (2x1024),
        # need to aggregate them
        if X.shape[0] > 1:
            X = np.mean(X, axis=0)
        else:
            X = X.flatten()


        embedding = torch.tensor(embedding, dtype=torch.float32)
        
        if self.transform:
            embedding = self.transform(embedding)
            
        # Encode the label as a plain Python int.
        label_encoded = int(self.label_encoder.transform([label])[0])
        
        return embedding, label_encoded

    def get_labels(self):
        """Return the encoded labels for the entire dataset."""
        return self.label_encoder.transform(self.data_frame["category"]).tolist()
    

def few_shot_dataloader(embed_dir, data_frame, n_way, n_shot, n_query, n_tasks, transform=None):
    """
    Args:
        embed_dir: Directory where embedding files are stored.
        data_frame: DataFrame (or path to CSV) with 'filename' and 'category' columns.
        n_way: Number of classes per task.
        n_shot: Number of support samples per class.
        n_query: Number of query samples per class.
        n_tasks: Number of few-shot tasks (episodes) per epoch.
        transform: Optional transform for the embeddings.
    """
    dataset = EmbeddingFileDataset(embed_dir, data_frame, transform=transform)
    
    sampler = TaskSampler(
        dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_tasks
    )
    
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        pin_memory=False,
        collate_fn=sampler.episodic_collate_fn  # Provided by easyfsl
    )
    
    return loader


class EmbeddingsMiniESC50DataModule(LightningDataModule):
    def __init__(
        self,
        embed_dir_train: str,
        embed_dir_val: str,
        embed_dir_test: str,
        csv_file_train: str,  
        csv_file_val: str,
        csv_file_test: str,
        n_task_train: int = 100,
        n_task_val: int = 20,
        n_task_test: int = 10,
        transform=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dir_train = embed_dir_train
        self.embed_dir_val = embed_dir_val
        self.embed_dir_test = embed_dir_test
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
        self.train_set = pd.read_csv(self.csv_file_train)
        self.val_set = pd.read_csv(self.csv_file_val)
        self.test_set = pd.read_csv(self.csv_file_test)

    def train_dataloader(self):
        return few_shot_dataloader(
            embed_dir=self.embed_dir_train,
            data_frame=self.train_set,
            n_way=5,
            n_shot=3,
            n_query=2,
            n_tasks=self.n_task_train,
            transform=self.transform
        )
    
    def val_dataloader(self):
        return few_shot_dataloader(
            embed_dir=self.embed_dir_val,
            data_frame=self.val_set,
            n_way=5,
            n_shot=3,
            n_query=2,
            n_tasks=self.n_task_val,
            transform=self.transform
        )
    
    def test_dataloader(self):
        return few_shot_dataloader(
            embed_dir=self.embed_dir_test,
            data_frame=self.test_set,
            n_way=5,
            n_shot=3,
            n_query=2,
            n_tasks=self.n_task_test,
            transform=self.transform
        )

    

        
