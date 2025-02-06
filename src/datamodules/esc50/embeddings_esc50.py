import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from easyfsl.samplers import TaskSampler
from pytorch_lightning import LightningDataModule
import pandas as pd

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_dict, data_frame, transform=None):
        """
        embeddings_dict: Dictionary mapping filename -> embedding (as a NumPy array)
        data_frame: Either a path to a CSV file or a Pandas DataFrame with columns 'filename' and 'category'
        transform: Optional transform to apply to each embedding
        """
        if isinstance(data_frame, str):
            self.data_frame = pd.read_csv(data_frame)
        else:
            self.data_frame = data_frame.reset_index(drop=True)
            
        self.embeddings_dict = embeddings_dict
        self.transform = transform

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data_frame["category"])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        filename = row["filename"]
        label = row["category"]

        embedding = self.embeddings_dict[filename]
        embedding = torch.tensor(embedding, dtype=torch.float32)

        if self.transform:
            embedding = self.transform(embedding)

        # Convert the label to an integer
        label_encoded = int(self.label_encoder.transform([label])[0])
        return embedding, label_encoded

    def get_labels(self):
        """Return a list of all encoded labels."""
        return self.label_encoder.transform(self.data_frame["category"]).tolist()
    

def few_shot_dataloader(embeddings_dict, data_frame, n_way, n_shot, n_query, n_tasks, transform=None):
    """
    embeddings_dict: Dictionary mapping filename -> embedding (NumPy array)
    data_frame: DataFrame (or path to CSV) with 'filename' and 'category' columns.
    n_way: Number of classes per task.
    n_shot: Number of support samples per class.
    n_query: Number of query samples per class.
    n_tasks: Number of few-shot tasks (episodes) per epoch.
    transform: Optional transform for the embeddings.
    """
    dataset = EmbeddingDataset(embeddings_dict, data_frame, transform=transform)
    
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
        collate_fn=sampler.episodic_collate_fn  # this collate function is provided by easyfsl.
    )
    
    return loader


class EmbeddingsMiniESC50DataModule(LightningDataModule):
    def __init__(
        self,
        embeddings_train: dict,
        embeddings_val: dict,
        embeddings_test: dict,
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
        self.embeddings_train = embeddings_train
        self.embeddings_val = embeddings_val
        self.embeddings_test = embeddings_test
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
        train_loader = few_shot_dataloader(
            self.embeddings_train,
            self.train_set,  
            n_way=5,
            n_shot=3,
            n_query=2,
            n_tasks=self.n_task_train,
            transform=self.transform
        )
        return train_loader
    
    def val_dataloader(self):
        val_loader = few_shot_dataloader(
            self.embeddings_val,
            self.val_set,  
            n_way=5,
            n_shot=3,
            n_query=2,
            n_tasks=self.n_task_val,
            transform=self.transform
        )
        return val_loader
    
    def test_dataloader(self):
        test_loader = few_shot_dataloader(
            self.embeddings_test,
            self.test_set,  
            n_way=5,
            n_shot=3,
            n_query=2,
            n_tasks=self.n_task_test,
            transform=self.transform
        )
        return test_loader
    

        
