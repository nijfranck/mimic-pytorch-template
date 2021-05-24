import torch
from PIL import Image

class MimicDataset():
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(self.df['class'][index])

        if self.transform:
            X = self.transform(X)

        return X, y