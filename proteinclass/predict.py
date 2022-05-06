############################################################################################
#                                                                                          #
#                                   Main prediction routine                                #
#                                                                                          #
#                                   Lionel Cheng, 05.05.2022                               #
#                                                                                          #
############################################################################################
# PyTorch
from typing import Sequence
import torch
from torch.utils.data import Dataset

# Others
from sklearn.preprocessing import LabelEncoder
import yaml
import argparse
import numpy as np
from pathlib import Path
import pandas as pd

# Internal routines
from .dataloader import integer_encoding, pad_sequences
from .model import SimpleProteinClass, LSTMProteinClass
from .log import create_log


class Predictor:
    """ Class for inference of given input sequences. """
    def __init__(self, model, label_encoder, seq_dataset, cfg):
        # Create logger for training
        self.cfg = cfg
        self.save_dir = Path(self.cfg['location']) / 'eval'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = create_log('predict', self.save_dir, logformat='small', console=True)

        # Copy configuration dictionnary in case folder
        with open(self.save_dir / 'config.yml', 'w') as file:
            yaml.dump(cfg, file)

        # Load the label encoder
        self.label_encoder = label_encoder

        # Store the sequences dataset
        self.seq_dataset = seq_dataset

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(self.cfg['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        # Resume training
        self._resume_checkpoint(self.cfg['location'] + '/model_best.pth')

    def predict(self):
        """ Predict the label of a list of sequences. """
        for idx, seq in enumerate(self.seq_dataset):
            seq = torch.from_numpy(seq)
            seq = seq.to(self.device)
            predicted_label_vec = self.model(seq)
            label = predicted_label_vec.argmax(1)
            family = self.label_encoder.classes_[label]
            self.logger.info(f'Sequence #{idx+1:d} belongs to {family} family')

    def _prepare_device(self, n_gpu_use):
        """ Setup GPU device if available, move model into configured device. """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _resume_checkpoint(self, resume_path):
        """ Resume from the given saved checkpoint. """
        resume_path = str(resume_path)
        self.logger.info('Loading checkpoint: {} ...'.format(resume_path))

        checkpoint = torch.load(resume_path)
        self.model.load_state_dict(checkpoint['state_dict'])

        self.logger.info('Checkpoint loaded.')

class SequenceDataset(Dataset):
    """Sequence dataset for reading PFAM dataset. """
    def __init__(self, sequences, max_length):
        # The sequences
        self.sequences = sequences
        self.n_seq = len(self.sequences)

        # The DataFrame is the filetered dataset
        self.df = pd.DataFrame(sequences, columns=['sequence'])

        # Encode the 'sequence' entry of the DataFrame
        # using only the 20 most common amino acids
        self.encode = integer_encoding(self.df)

        # Convert all the dataset to a fixed length by padding
        # or cutting the longer/smaller sequences
        self.max_length = max_length
        self.dataset_pad = pad_sequences(self.encode, maxlen=max_length,
                                         padding='post', truncating='post')

    def __len__(self):
        return len(self.dataset_pad)

    def __getitem__(self, idx):
        return self.dataset_pad[idx, :]


def run_predict(cfg):
    """ Run training based on configuration dictionnary. """
    # Parse the configuration dicionnary
    data_path = Path(cfg['location'])
    n_mc = cfg['n_labels']
    max_length = cfg['max_length']

    # Create dataset objects
    le = LabelEncoder()
    le.classes_ = np.load(data_path / 'le_classes.npy', allow_pickle=True)

    # Creation of the training related objects
    # First the model
    model_cfg = cfg['model']
    embed_dim = model_cfg['args']['embed_dim']
    if model_cfg['type'] == 'SimpleProteinClass':
        model = SimpleProteinClass(21, embed_dim, max_length, n_mc)
    elif model_cfg['type'] == 'LSTMProteinClass':
        hidden_size = model_cfg['args']['hidden_size']
        model = LSTMProteinClass(21, embed_dim, max_length, n_mc, hidden_size)

    # Creatino of dataset from the sequence list
    sequences = cfg['sequences']
    seq_dataset = SequenceDataset(sequences, max_length)

    # Creation of trainer object
    predictor = Predictor(model, le, seq_dataset, cfg)

    # Training step
    predictor.predict()

def main():
    """ Wrapper around training routine.
    Parse the configuration dictionnary. """
    # Open configuration dictionnary
    parser = argparse.ArgumentParser(
        description='Training of Protein classification model')
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    # Run prediction
    run_predict(cfg)

if __name__ == '__main__':
    main()