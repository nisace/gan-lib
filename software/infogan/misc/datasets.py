import cPickle as pkl
import os

import numpy as np
from tensorflow.examples.tutorials import mnist

from utils.file_system_utils import download, untar


class Dataset(object):
    def __init__(self, images, labels=None):
        """
        Args:
            images (ndarray): shape (n, ...)
            labels (ndarray): shape (n,)
        """
        self._images = images.reshape(images.shape[0], -1)
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = images.shape[0]
        # shuffle on first run
        self._index_in_epoch = self._num_examples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            if self._labels is not None:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if self._labels is None:
            return self._images[start:end], None
        else:
            return self._images[start:end], self._labels[start:end]


class MnistDataset(object):
    def __init__(self):
        data_directory = "MNIST"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class Cifar10Dataset(object):
    def __init__(self):
        x_train, y_train = Cifar10Dataset.load_data()
        self.train = Dataset(x_train, y_train)
        self.image_dim = 32 * 32 * 3
        self.image_shape = (32, 32, 3)

    @staticmethod
    def load_data():
        origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        origin_file_name = os.path.basename(origin)
        download_folder = 'CIFAR-10'
        download_path = os.path.join(download_folder, origin_file_name)
        download(origin, download_path)
        untar_path = os.path.join(download_folder, 'cifar-10-batches-py')
        untar(download_path, download_folder, untar_path)
        x_train = []
        y_train = []
        for i in range(1, 6):
            batch_path = os.path.join(untar_path, 'data_batch_{}'.format(i))
            data, labels = Cifar10Dataset.load_batch(batch_path)
            x_train.append(data)
            y_train.append(labels)
        return np.concatenate(x_train), np.concatenate(y_train)

    @staticmethod
    def load_batch(batch_file_path):
        with open(batch_file_path, 'r') as f:
            d = pkl.load(f)
        data = d['data']
        data = data.reshape(data.shape[0], 3, 32, 32)  # (n, c, h, w)
        data = np.transpose(data, (0, 2, 3, 1))  # (n, h, w, c)
        return data, d['labels']

    def inverse_transform(self, data):
        return data
