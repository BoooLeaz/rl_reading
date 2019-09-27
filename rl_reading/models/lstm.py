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
        self.convnet_output_size = 120
        self.encoder_gru = torch.nn.GRU(input_size=self.convnet_output_size,
                                        hidden_size=self.gru_hidden_size,
                                        num_layers=1,
                                        batch_first=True)

    def forward(self, x, h):
        """
        :param x: torch tensor (sequence_length, width, height), dtype: float32
        :param h: (num_rnn_directions * rnn_layers, batch_size, hidden_size)
            in our case: (1, 1, hidden_size)

        :returns hidden: (num_rnn_directions * rnn_layers, batch_size, hidden_size)
        """
        sequence_length = x.size(0)
        # in the batch dimension of the convnet we put the sequence dimension instead
        # in this way we can only have batch_size 1
        x = self.convnet(x)
        # reshape to (batch_size=1, sequence_length, gru_input_size)
        x = x.view(1, sequence_length, self.convnet_output_size)
        _, h = self.encoder_gru(x, h)
        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        return h


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
        :param x: tensor shape (,), dtype: int64
            contains the last predicted digit
        :param h: tensor shape (1, 1, gru_hidden_size)
            contains the hidden state of the RNN.
        """
        # add sequence_length and num_rnn_directions dimensions
        x = x.view(1, 1)
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

    def forward(self, x, y):
        """
        :param x: (batch_size=1, channels, width, height)
        :param y: (target_sequence_length)
            consists of integers indexing correct characters

        :returns outputs: torch tensor (target_sequence_length, decoder.n_actions), dtype: float32
        """
        target_sequence_length = y.shape[0]

        # get patches
        x = torch.transpose(torch.nn.Unfold(kernel_size=(32, 32), stride=(32, 32))(x), 1, 2).view(-1, 1, 32, 32)
        # x shape: (sequence_length, channels, width, height)

        #tensor to store decoder outputs
        outputs = torch.zeros(target_sequence_length, self.decoder.n_actions).to(self.device)

        # initialize hidden (batch_size, num_rnn_directions, hidden_size)
        hidden = torch.zeros(1, 1, self.encoder.gru_hidden_size)

        # first input to the decoder
        pred_char = torch.zeros(size=(1,) , dtype=torch.int64)

        for t in range(0, target_sequence_length):
            # last hidden state of the encoder is used as the initial hidden state of the decoder
            hidden = self.encoder(x[[t]], hidden)
            # hidden: (num_rnn_directions * rnn_layers, batch_size, hidden_size)

            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.decoder(pred_char, hidden)
            # output (batch_size, sequence_length, num_rnn_directions * hidden_size)
            # hidden (batch_size, num_rnn_directions, hidden_size)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #get the highest predicted token from our predictions
            top1 = output.argmax(axis=2).squeeze()
            # top1 shape (,)

            # next input
            pred_char = y[t] if random.random() < 0.5 else top1
        return outputs
