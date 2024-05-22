import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def encode(s, vocabulary):

    s_to_i = {ch:i for i,ch in enumerate(vocabulary)}
    return torch.tensor([s_to_i[c] for c in s]).long()

def decode(s, vocabulary):

    i_to_s = {i:ch for i,ch in enumerate(vocabulary)}
    return ''.join([i_to_s[i.item()] for i in s])


class TinyShakespeare(Dataset):
    """
    Implement Shakespeare's text as a PyTorch dataset.
    """

    def __init__(
            self,
            path,
            vocab_size,
            block_size,
            seed_sample,
            train_size,
            test_size=32768,
            max_train_size=1115394-524288-32768,
            whitening=0,            
            transform=None,
    ):

        self.vocab_size = vocab_size
        self.block_size = block_size

        with open(path + 'tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        self.vocabulary = sorted(list(set(text)))
        assert len(self.vocabulary)==vocab_size, f'vocab_size ({vocab_size}) does not match actual vocabulary'
        text = encode(text, self.vocabulary)

        random.seed(seed_sample)
        samples_train = torch.tensor( random.sample( range(max_train_size+1-block_size), train_size))
        samples_test = torch.tensor( random.sample( range(max_train_size,1115394+1-block_size), test_size))
        samples = torch.cat((samples_train,samples_test),dim=0)
        
        self.features = torch.stack([text[i:i+self.block_size] for i in samples])
        self.labels = self.features[:,-1]

        self.features = F.one_hot(
            self.features,
            num_classes=vocab_size
        ).float()
    
        if whitening:

            inv_sqrt_norm = (1.-1./vocab_size) ** -.5
            self.features = (self.features - 1./vocab_size) * inv_sqrt_norm

        self.features = self.features.permute(0,2,1)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
        	idx: sample index

        Returns:
            Feature-label pairs at index            
        """
        x, y = self.features[idx], self.labels[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y
