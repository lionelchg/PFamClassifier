############################################################################################
#                                                                                          #
#                                    Main training routine                                 #
#                                                                                          #
#                                   Lionel Cheng, 03.05.2022                               #
#                                                                                          #
############################################################################################
# PyTorch
import torch
from torch.utils.data import DataLoader

# Others
from sklearn.preprocessing import LabelEncoder
import yaml
import argparse

# Internal routines
from .dataloader import ProteinDataset
from .model import SimpleProteinClass, LSTMProteinClass, RNNProteinClass
from .trainer import Trainer

def run_train(cfg):
    """ Run training based on configuration dictionnary. """
    # Parse the configuration dicionnary
    data_path = cfg['data_path']
    batch_size = cfg['batch_size']
    n_mc = cfg['n_labels']
    max_length = cfg['max_length']

    # Create dataset objects
    le = LabelEncoder()
    train_dataset = ProteinDataset(data_path, 'train', n_mc, max_length, le)
    val_dataset = ProteinDataset(data_path, 'dev', n_mc, max_length,
                                le, commonLables=train_dataset.commonLables)
    test_dataset = ProteinDataset(data_path, 'test', n_mc, max_length,
                                le, commonLables=train_dataset.commonLables)

    # Create dataloader objects
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Creation of the training related objects
    # First the model
    model_cfg = cfg['model']
    embed_dim = model_cfg['args']['embed_dim']
    if model_cfg['type'] == 'SimpleProteinClass':
        model = SimpleProteinClass(21, embed_dim, max_length, n_mc)
    elif model_cfg['type'] == 'LSTMProteinClass':
        num_layers = model_cfg['args']['num_layers']
        hidden_size = model_cfg['args']['hidden_size']
        model = LSTMProteinClass(21, embed_dim, max_length, n_mc,
            num_layers, hidden_size)
    elif model_cfg['type'] == 'RNNProteinClass':
        num_layers = model_cfg['args']['num_layers']
        hidden_size = model_cfg['args']['hidden_size']
        model = RNNProteinClass(21, embed_dim, max_length, n_mc,
            num_layers, hidden_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, cfg['optimizer']['type'])(
        model.parameters(), **cfg['optimizer']['args'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **cfg['scheduler'])

    # Creation of trainer object
    trainer = Trainer(model, criterion, optimizer,
                train_dataloader, val_dataloader, test_dataloader, le,
                scheduler, cfg)

    # Training step
    trainer.train()

    # Evaluate the model on the test dataset at the end of training
    trainer.test()

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

    # Run training
    run_train(cfg)

if __name__ == '__main__':
    main()