import torch
from collections import OrderedDict

from . import basemodel


class Model(basemodel.BaseModel):
    def __init__(self, params, actions, name=None):
        super(Model, self).__init__()

        self.gru_hidden = params['gru_hidden']  # number of hidden neurons in each layer
        self.n_actions = len(actions)

        self.batch_size = 1
        self._initialize()

    def _initialize(self):
        self.convnet = torch.nn.Sequential(OrderedDict([
            ('c1', torch.nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', torch.nn.ReLU()),
            ('s2', torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', torch.nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', torch.nn.ReLU()),
            ('s4', torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', torch.nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', torch.nn.ReLU())
        ]))
        # TODO input size
        self.encoder_gru = torch.nn.GRU(input_size=self.gru_hidden,
                                          hidden_size=self.gru_hidden,
                                          num_layers=1,
                                          batch_first=True)

    def forward(self, x):
        """
        x: (sequence_length, width, height)
        """
        sequence_length = x.size(0)
        x = x.contiguous()
        # in the batch dimension of the convnet we put the sequence dimension instead
        # in this way we can only have batch_size 1
        x = self.convnet(x)
        # reshape to (batch_size, sequence_length, gru_input_size)
        x = x.view(1, sequence_length, -1)
        # debug: get x.shape[-1] for gru input size
        import ipdb; ipdb.set_trace()

        self.reset()
        _, h_n = self.encoder_gru(x)
        # h_n shape: (num_layers * num_directions, batch, hidden_size)


        q = v + a - torch.mean(a, keepdim=True, dim=-1)
        # Q: batch x n_actions
        # V: batch x 1
        # A: batch x n_actions
        # Q(state, action) = V(state) + A(state, action) [- 1/n_action sum_action A(state, action)]
        return q, v, a
