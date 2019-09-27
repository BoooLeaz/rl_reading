import torch


def get_reward(y_hat, y):
    """
    :param y_hat: shape (sequence_length,), dtype int64
    :param y: shape (sequence_length,), dtype int64
    """
    r = torch.zeros(size=(y_hat.shape[0],), dtype=torch.float32)

    for i in range(y_hat.shape[0]):
        if y_hat[i] == y[i]:
            r[i] = 1
        else:
            r[i] = -1
    return r

