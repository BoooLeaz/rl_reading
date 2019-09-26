import torch
from collections import OrderedDict
import random

from . import basemodel


def init_weights(m):
    for name, param in m.named_parameters():
        torch.nn.init.uniform_(param.data, -0.08, 0.08)


class Encoder(basemodel.BaseModel):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.gru_hidden_size = params['gru_hidden_size']
        self._initialize()

    def _initialize(self):
        self.convnet = torch.nn.Sequential(OrderedDict([
            ('c1', torch.nn.Conv2d(1, 6, kernel_size=(5, 5), padding=(0, 0))),
            ('relu1', torch.nn.ReLU()),
            ('s2', torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', torch.nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', torch.nn.ReLU()),
            ('s4', torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', torch.nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', torch.nn.ReLU())
        ]))
        self.encoder_gru = torch.nn.GRU(input_size=self.gru_hidden_size,
                                        hidden_size=self.gru_hidden_size,
                                        num_layers=1,
                                        batch_first=True)

    def forward(self, x):
        """
        :param x: torch tensor (sequence_length, width, height), dtype: float32
        """
        sequence_length = x.size(0)
        # in the batch dimension of the convnet we put the sequence dimension instead
        # in this way we can only have batch_size 1
        x = self.convnet(x)
        # reshape to (batch_size=1, sequence_length, gru_input_size)
        x = x.view(1, sequence_length, self.gru_hidden_size)
        _, hidden = self.encoder_gru(x)
        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        return hidden


class Decoder(basemodel.BaseModel):
    def __init__(self, params, n_actions, n_characters):
        super(Decoder, self).__init__()

        self.gru_hidden_size = params['gru_hidden_size']  # number of hidden neurons in each layer
        self.n_actions = n_actions
        self.n_characters = n_characters
        self.action_encodings = torch.eye(self.n_actions, self.n_actions)
        self._initialize()

    def _initialize(self):
        self.decoder_gru = torch.nn.GRU(input_size=self.n_characters,
                                        hidden_size=self.gru_hidden_size,
                                        num_layers=1,
                                        batch_first=True)
        self.out = torch.nn.Linear(in_features=self.gru_hidden_size, out_features=self.n_actions)

    def forward(self, x, h):
        """
        :param x: tensor shape (1,), dtype: int64
            contains the last predicted digit
        :param h: tensor shape (1, gru_hidden_size)
            contains the hidden state of the RNN.
        """
        # add batch dimension
        x = x.unsqueeze(0)
        # one-hot encode the input characters
        one_hot_embedded = torch.nn.functional.one_hot(x, self.n_characters).to(torch.float32)
        # do one step with the RNN (output and hidden state of the rnn are the same here)
        _, h = self.decoder_gru(one_hot_embedded, h)
        # use the RNN hidden state to compute an output indicating which character comes next
        output = self.out(h)
        return output, h


class EncoderDecoder(basemodel.BaseModel):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, x, y, teacher_forcing_ratio=0.5):
        """
        :param x: (input_sequence_length, channels, width, height)
        :param y: (target_sequence_length)
            consists of integers indexing correct characters

        :returns outputs: torch tensor (target_sequence_length, decoder.n_actions), dtype: float32
        """
        target_sequence_length = y.shape[0]

        #tensor to store decoder outputs
        outputs = torch.zeros(target_sequence_length, self.decoder.n_actions).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(x)

        # first input to the decoder is the <sos> tokens
        x = torch.zeros(size=(1,) , dtype=torch.int64)

        for t in range(0, target_sequence_length):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.decoder(x, hidden)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            top1 = output.argmax(axis=1)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            x = y[t] if teacher_force else top1
        return outputs
