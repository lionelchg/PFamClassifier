############################################################################################
#                                                                                          #
#                                        Model classes                                     #
#                                                                                          #
#                                   Lionel Cheng, 03.05.2022                               #
#                                                                                          #
############################################################################################
import torch
from torch import nn
from abc import abstractmethod
import numpy as np

class BaseModel(nn.Module):
    """
    Base class for all models. Overrides __str__ method and forces reimplementation of forward.
    """
    @abstractmethod
    def forward(self, *inputs):
        """ Forward pass logic """
        raise NotImplementedError

    @property
    def nparams(self):
        """ Number of trainable parameters of the model"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def __str__(self):
        """ Models prints with number of trainable parameters """
        return super().__str__() + '\nTrainable parameters: {}'.format(self.nparams)


class SimpleProteinClass(BaseModel):
    """ Simple Protein classifier using an embedding layer
    followed by a Fully Connected layer.
    """
    def __init__(self, vocab_size, embed_dim, max_length, num_class):
        super(SimpleProteinClass, self).__init__()
        # Store global class properties
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.num_class = num_class

        # Build the network layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim * max_length, num_class)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded_reshaped = torch.reshape(embedded, (-1, self.embed_dim * self.max_length))
        return self.fc(embedded_reshaped)

class LSTMProteinClass(BaseModel):
    """ Protein classifier using an embedding layer
    followed by LSTM layers after a final fully Connected layer.
    """
    def __init__(self, vocab_size, embed_dim, max_length,
                num_class, num_layers, hidden_size):
        super(LSTMProteinClass, self).__init__()
        # Store global class properties
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.num_class = num_class
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Build the network layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim,
                           hidden_size=hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True,
                           bidirectional=True)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        x = self.embedding(x)
        x = self.drop(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.fc(ht[-1])

class RNNProteinClass(BaseModel):
    """ Protein classifier using an embedding layer
    followed by RNN layers after a final fully Connected layer.
    """
    def __init__(self, vocab_size, embed_dim, max_length,
                num_class, num_layers, hidden_size):
        super(RNNProteinClass, self).__init__()
        # Store global class properties
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.num_class = num_class
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Build the network layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim,
                           hidden_size=hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        x = self.embedding(x)
        rnn_out, ht = self.rnn(x)
        return self.fc(ht[-1])