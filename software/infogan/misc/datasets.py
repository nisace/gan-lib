import os
from glob import glob

import cPickle as pkl
import numpy as np
from tensorflow.examples.tutorials import mnist

from utils.file_system_utils import download, extract_all
from utils.image_utils import get_image

DATA_FOLDER = 'data'


class DatasetIterator(object):
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

    def transform(self, data):
        return data

    def transform_batch(self, batch):
        return batch

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
        data = self._images[start:end]
        data = [self.transform(d) for d in data]
        data = self.transform_batch(data)
        if self._labels is None:
            return data, None
        else:
            return data, self._labels[start:end]


class CelebADatasetIterator(DatasetIterator):
    def transform(self, data):
        return get_image(str(data[0]), 108, 108,
                         resize_height=64, resize_width=64,
                         is_crop=True, is_grayscale=False)

    def transform_batch(self, batch):
        return np.array(batch).reshape(-1, 64 * 64 * 3)


class MnistDataset(object):
    def __init__(self):
        data_directory = os.path.join(DATA_FOLDER, "MNIST")
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
        self.supervised_train = DatasetIterator(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def __getstate__(self):
        return {}

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class Cifar10Dataset(object):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
        x_train, y_train = self.load_data()
        self.train = DatasetIterator(x_train, y_train)
        self.image_dim = 32 * 32 * 3
        self.image_shape = (32, 32, 3)

    def load_data(self):
        origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        origin_file_name = os.path.basename(origin)
        download_folder = os.path.join(DATA_FOLDER, 'CIFAR-10')
        download_path = os.path.join(download_folder, origin_file_name)
        download(origin, download_path)
        extract_path = extract_all(download_path)
        x_train = []
        y_train = []
        for i in range(1, 6):
            batch_path = os.path.join(extract_path, 'data_batch_{}'.format(i))
            data, labels = self.load_batch(batch_path)
            x_train.append(data)
            y_train.append(labels)
        x_train = np.concatenate(x_train)
        x_train = x_train / 127.5 - 1.
        y_train = np.concatenate(y_train)
        return x_train, y_train

    def load_batch(self, batch_file_path):
        with open(batch_file_path, 'r') as f:
            d = pkl.load(f)
        data = d['data']
        data = data.astype(self.dtype)
        data = data.reshape(data.shape[0], 3, 32, 32)  # (n, c, h, w)
        data = np.transpose(data, (0, 2, 3, 1))  # (n, h, w, c)
        return data, d['labels']

    def inverse_transform(self, data):
        return data


class CelebADataset(object):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
        self.train = CelebADatasetIterator(self.images_paths())
        self.image_dim = 64 * 64 * 3
        self.image_shape = (64, 64, 3)

    def images_paths(self):
        origin = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1&pv=1'
        origin_file_name = os.path.basename(origin).split('?')[0]
        download_folder = os.path.join(DATA_FOLDER, 'celebA')
        download_path = os.path.join(download_folder, origin_file_name)
        download(origin, download_path)
        extract_path = extract_all(download_path)
        return np.array(glob(os.path.join(extract_path, '*.jpg')))

    def inverse_transform(self, data):
        return data


class Horse2Zebra(object):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
