import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from sklearn.model_selection import train_test_split


import email, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


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
    print('shape: ', df.shape)
    df = df.drop('index', axis=1)
    df = df.reset_index(drop=True)
    print(df.head())
    non_pneumonia = df[df['Pneumonia'] == 0.0].reset_index(drop=True)
    pneumonia = df[df['Pneumonia'] == 1.0].reset_index(drop=True)
    result = pd.concat([non_pneumonia, pneumonia], ignore_index=True, sort=False)
    result['class'] = result['Pneumonia']
    result['class'] = result['class'].apply(lambda x: 1 if x == 1.0 else 0)
    result = result.reset_index(drop=True)
    # training, test set
    y = result['class']
    data_train, data_test = train_test_split(result, test_size=0.1, random_state=101, stratify=y)
    print('data_train', data_train.shape)
    print('data_test', data_test.shape)
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)
    data_train.to_csv('data/train.csv', index=False)
    data_test.to_csv('data/test.csv', index=False)
    print('Finished saving train/test validation')

def load_csv(csv_filename):
    arr = os.listdir('data')
    df = pd.read_csv('data/' + csv_filename)
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


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,
                          save_directory=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_directory)

def send_email(sender_email, receiver_email, subject, filename):

    print(filename)
    message = MIMEMultipart()
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = receiver_email
    body = "This is an email with attachment of your recent GPU run"
    password = os.environ.get('password')

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    filename = filename  # In same directory as script

    # Open PDF file in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)


    # Create the plain-text and HTML version of your message
    text = """\
    Hi,
    How are you?
    Real Python has many great tutorials:
    www.realpython.com"""
    html = """\
    <html>
      <body>
        <p>Hi,<br>
           How are you?<br>
           <a href="http://www.realpython.com">Real Python</a> 
           has many great tutorials.
        </p>
      </body>
    </html>
    """

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
