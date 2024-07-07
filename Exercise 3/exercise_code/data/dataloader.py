"""Definition of Dataloader"""

import numpy as np


def build_batches(dataset, batch_size):
    batches = []  # list of all mini-batches
    batch = []  # current mini-batch
    for i in range(len(dataset)):
        batch.append(dataset[i])
        if len(batch) == batch_size:  # if the current mini-batch is full,
            batches.append(batch)  # add it to the list of mini-batches,
            batch = []  # and start a new mini-batch
    return batches


def combine_batch_dicts(batch):
    batch_dict = {}
    for data_dict in batch:
        for key, value in data_dict.items():
            if key not in batch_dict:
                batch_dict[key] = []
            batch_dict[key].append(value)
    return batch_dict


def batch_to_numpy(batch):
    numpy_batch = {}
    for key, value in batch.items():
        numpy_batch[key] = np.array(value)
    return numpy_batch


def build_batch_iterator(dataset, batch_size, shuffle, drop_last):
    if shuffle:
        index_iterator = iter(np.random.permutation(len(dataset)))
    else:
        index_iterator = iter(range(len(dataset)))

    batch = []
    for index in index_iterator:
        batch.append(dataset[index])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if not drop_last and len(batch) >= 0:
        yield batch


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        index_iterator = build_batch_iterator(
            self.dataset, self.batch_size, self.shuffle, self.drop_last)

        for batch in index_iterator:
            batch_dict = combine_batch_dicts(batch)
            numpy_batch = batch_to_numpy(batch_dict)
            yield numpy_batch

    def __len__(self):
        if self.drop_last:
            length = len(self.dataset) // self.batch_size
        else:
            length = len(self.dataset) // self.batch_size + \
                (len(self.dataset) % self.batch_size != 0)

        return length
