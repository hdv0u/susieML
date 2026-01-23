import torch
from torch.utils.data import Dataset

# debug: print(f"debug: widget {__name__}")

class NumpyDataset(Dataset):
    def __init__(self, X, y, multi_class=False) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long if multi_class else torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]