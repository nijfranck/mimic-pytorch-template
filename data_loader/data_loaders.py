from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader import MimicDataset
from utils import load_csv

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MimicPneumoniaDataloader(BaseDataLoader):
    """
    MIMIC-CXR-JPG data loading
    """
    def __init__(self, csv_name, batch_size, shuffle=True, validation_split=0.1, num_workers=4, training=True, input_size=200,
                 norm_mean=(0.5, ), norm_std=(0.5, )):
        trsfm = transforms.Compose([transforms.Resize((input_size,input_size)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
        self.data_frame = load_csv(csv_name)
        self.dataset = MimicDataset(self.data_frame, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)