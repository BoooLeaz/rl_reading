import torch
import os


class BaseModel(torch.nn.Module):
    def __init__(self):
        """__init__

        :param n_actions: int
        :param n_hidden: int, number of hidden layers
        :param name:
        :param feature_transformation: callable or None,
            if callable: callable with input array-like of shape (n_batch, 3)
                should return array like of same shape
            if None uses identity transformation
        """
        super(BaseModel, self).__init__()

    def _initialize(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def predict(self, x):
        x = torch.from_numpy(x).float()
        with torch.set_grad_enabled(False):
            pytorch_results = self.forward(x)

        # extract into numpy
        # if there are multiple return values, we need to
        # extract each one separately
        if type(pytorch_results) is tuple:
            results = []
            for result in pytorch_results:
                results.append(result.numpy())
        else:
            results = pytorch_results.numpy()
        return results

    def reset(self):
        """reset
        Used for instance to reset recurrent networks. Inheriting models don't need to
        override this if not needed."""
        pass

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self, path)
