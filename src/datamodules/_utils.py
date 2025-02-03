from abc import abstractmethod

import random
import librosa
import os

from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Iterator

import torch
from torch import Tensor
from torch.utils.data import Sampler, Dataset

def to_mel_spectrogram(
        sig,
        sr,
        n_fft = 2048,
        hop_length = 512
        n_mel = 128):
    
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=sig, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mel
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db


class AudioDataset(Dataset):
    """
    AudioDataset assumes that the audio files are stored in a specific folder
    and the list of labels is stored in a CSV file in the column "category"
    """

    def __init__(self, root_dir, data_frame, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_frame = data_frame

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data_frame["category"])

    def __len__(self):
        return len(self.data_frame)

    def get_labels(self):
        labels = []

        for i in range(0, len(self.data_frame)):
            label = self.data_frame.iloc[i]["category"]
            label = self.label_encoder.transform([label])[0]
            labels.append(label)

        return labels

    def __getitem__(self, idx):
        audio_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]["filename"])
        label = self.data_frame.iloc[idx]["category"]

        # Load audio data and perform any desired transformations
        sig, sr = librosa.load(audio_path, sr=16000, mono=True)
        sig_t = torch.tensor(sig)
        # padding_mask = torch.zeros(1, sig_t.shape[0]).bool().squeeze(0)
        if self.transform:
            sig_t = self.transform(sig_t)

        # Encode label as integer
        label = self.label_encoder.transform([label])[0]

        return sig_t, label

