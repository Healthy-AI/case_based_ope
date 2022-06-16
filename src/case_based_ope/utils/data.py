import skorch
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.model_selection import ShuffleSplit

def uses_placeholder_y(ds):
    if isinstance(ds, torch.utils.data.Subset):
        return uses_placeholder_y(ds.dataset)
    return isinstance(ds, skorch.dataset.Dataset) and hasattr(ds, "y") and ds.y is None

class InternalSplit:
    def __init__(self, train_size=0.8, random_state=None):
        self.train_size = train_size
        self.random_state = random_state if random_state is not None else np.random.randint(0, 101)  # Enables reproducing the split
    
    def __call__(self, dataset, _):
        splitter = ShuffleSplit(n_splits=1, train_size=self.train_size, random_state=self.random_state)
        args = (np.arange(dataset.__len__()),)
        ii_train, ii_valid = next(splitter.split(*args))
        dataset_train = torch.utils.data.Subset(dataset, ii_train)
        dataset_valid = torch.utils.data.Subset(dataset, ii_valid)
        return dataset_train, dataset_valid

class StandardDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def _transform(self, X, y):
        X = torch.Tensor(X)
        if y is None:
            y = torch.Tensor([0])
        else:
            pass
        return X, y

    def __getitem__(self, i):
        X, y = self.X, self.y
        Xi = X[i]
        yi = y[i] if y is not None else y
        return self._transform(Xi, yi)

    def __len__(self):
        return len(self.X)

class SequentialDataset(Dataset):
    def __init__(self, X, y=None):
        '''
        Parameters
        ----------
        X : NumPy Array of shape (n_samples, n_features)
            Input data. The last column of X should represent the group.
        y : NumPy Array of shape (n_samples,)
            Output targets.
        '''
        assert isinstance(X, np.ndarray)
        if y is not None: assert isinstance(y, np.ndarray)
        super(Dataset, self).__init__()
        sequences = []
        sequence_targets = [] if y is not None else y
        Xg = pd.DataFrame(X)
        c_group = Xg.columns[-1]
        for _, sequence in Xg.groupby(by=c_group):
            sequences += [sequence.drop(c_group, axis=1)]  # Exclude the groups
            ii_sequence = list(sequence.index)
            if y is not None: sequence_targets += [y[ii_sequence]]
        self.sequences = sequences
        self.sequence_targets = sequence_targets

    def _transform(self, X, y):
        X = torch.Tensor(X.values)
        if y is None:
            y = torch.Tensor([0])
        else:
            assert isinstance(y,  np.ndarray)
            y = torch.Tensor(y).type(torch.LongTensor)
        return X, y

    def __getitem__(self, i):
        X, y = self.sequences, self.sequence_targets
        Xi = X[i]
        yi = y[i] if y is not None else y
        return self._transform(Xi, yi)

    def __len__(self):
        return len(self.sequences)

def pad_pack_sequences(batch):
    sequences, targets = zip(*batch)
    lengths = [sequence.shape[0] for sequence in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)
    packed_padded_sequences = pack_padded_sequence(
        padded_sequences,
        batch_first=True,
        lengths=lengths,
        enforce_sorted=False
    )
    targets = torch.cat(targets, dim=0)
    return packed_padded_sequences, targets
