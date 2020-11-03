import torch
from torch.utils.data import Dataset

class MemexQA(Dataset):
    def __init__(self, questions, albums):
        self.questions = questions
        self.albums = albums
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):