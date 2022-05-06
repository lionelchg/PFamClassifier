# Reading the database
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# Creating a unique int for each label
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length. Taken from Keras 2.8.0.
    https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

# creating encoding dict for the common amino acids
codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def create_dict(codes):
    """ Convert codes list to dict. """
    char_dict = {}
    for index, val in enumerate(codes):
        char_dict[val] = index + 1

    return char_dict

char_dict = create_dict(codes)

#encoding the sequences using only the common amino acids dict
def integer_encoding(data):
    """
    - Encodes code sequence to integer values.
    - 20 common amino acids are taken into consideration
    and rest 4 are categorized as 0.
    """

    encode_list = []
    for row in data['sequence'].values:
        row_encode = []
        for code in row:
            row_encode.append(char_dict.get(code, 0))
        encode_list.append(np.array(row_encode))

    return encode_list

def read_data(data_path, partition):
    """ Read the raw csv data from the PFam dataset. """
    data = []
    for fn in os.listdir(os.path.join(data_path, partition)):
        with open(os.path.join(data_path, partition, fn)) as f:
            data.append(pd.read_csv(f, index_col=None))
    return pd.concat(data)

class ProteinDataset(Dataset):
    """Protein dataset for reading PFAM dataset. """
    def __init__(self, data_path: str, partition: str, n_mc: int, max_length: int,
                le, commonLables=None):
        """ Initialization of ProteinDataset

        :param data_path: Directory of the PFam dataset
        :type data_path: str
        :param partition: partition considered (either train/dev/test)
        :type partition: str
        :param n_mc: Number of most-common labels considered
        :type n_mc: int
        :param max_length: Maximum length of the sequences
        :type max_length: int
        :param le: Encoder for the family labels
        :type le: LabelEncoder
        :param commonLables: the common labels considered (for dev and test datasets)
        :type commonLables: _type_, optional
        """
        # Load the dataset as a pandas DataFrame
        df_tmp = read_data(data_path, partition)
        # selecting the n_mc most common labels
        self.n_mc = n_mc
        if isinstance(commonLables, pd.Series):
            self.commonLables = commonLables
        else:
            self.commonLables = df_tmp.family_accession.value_counts()[:self.n_mc]
        mask = df_tmp.family_accession.isin(self.commonLables.index.values)

        # The DataFrame is the filetered dataset
        self.df = df_tmp.loc[mask, :]
        # Encode the 'sequence' entry of the DataFrame
        # using only the 20 most common amino acids
        self.encode = integer_encoding(self.df)

        # Convert all the dataset to a fixed length by padding
        # or cutting the longer/smaller sequences
        self.max_length = max_length
        self.dataset_pad = pad_sequences(self.encode, maxlen=max_length,
                                         padding='post', truncating='post')

        # Create encoding for the labels
        if partition == 'train':
            self.y = le.fit_transform(self.df['family_accession'])
        else:
            self.y = le.transform(self.df['family_accession'])

    def __len__(self):
        return len(self.dataset_pad)

    def __getitem__(self, idx):
        return self.dataset_pad[idx, :], self.y[idx]
