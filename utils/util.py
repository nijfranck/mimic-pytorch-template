import json
import os
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from sklearn.model_selection import train_test_split

def split_train_test(csv_filename, positions='PA'):
    """
    Load MIMIC and decouple it into train and test, then save it to a data folder.
    TODO: Allow for multiple positions to be selected
    :param csv_filename:
    :param positions:
    :return:
    """
    print(csv_filename)
    df = pd.read_csv(csv_filename)
    print(df.head())
    df = df.drop('index', axis=1)
    df = df.reset_index(drop=True)
    print(df.head())
    df = df[(df['ViewPosition'] == positions)]
    non_pneumonia = df[df['Pneumonia'] == 0.0].reset_index(drop=True)
    pneumonia = df[df['Pneumonia'] == 1.0].reset_index(drop=True)
    result = pd.concat([non_pneumonia, pneumonia], ignore_index=True, sort=False)
    result['class'] = result['Pneumonia']
    result['class'] = result['class'].apply(lambda x: 1 if x == 1.0 else 0)
    result = result.reset_index(drop=True)
    # training, test set
    y = result['class']
    data_train, data_test = train_test_split(result, test_size=0.1, random_state=101, stratify=y)
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)
    data_train.to_csv('data/train.csv', index=False)
    data_test.to_csv('data/test.csv', index=False)
    print('Finished saving train/test validation')

def load_csv(csv_filename):
    arr = os.listdir('data')
    print(arr)
    df = pd.read_csv('data/' + csv_filename)
    print(df.head())
    return df


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(gpu_number, n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:'+ str(gpu_number) if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)