import argparse
import logging
import os
import sys
from ruamel.yaml import YAML
import torch
import sklearn.metrics

# imports from this project
import reward
import models
import mnist_david

yaml = YAML()
logger = logging.Logger('main_logger')


def main(params):
    output_path = params['output_path']
    device = 'cpu'

    with open(os.path.join(output_path, 'parameters.yaml'), 'w') as f:
        yaml.dump(params, f)

    # create results directories
    model_dir = os.path.join(output_path, 'models')
    data_basepath = os.path.join(output_path, 'data')
    for _path in [model_dir, data_basepath]:
        if not os.path.exists(_path):
            os.makedirs(_path)

    n_rows = 5
    n_cols = 5
    batch_size = 1
    train_size = 10000000000
    n_classes = 9
    clip_grad = 1

    # setup models
    # QModels
    model_module = getattr(models, params['model'])
    encoder = model_module.Encoder(params)
    decoder = model_module.Decoder(params, n_actions=n_classes, n_characters=n_classes)
    model = model_module.EncoderDecoder(encoder, decoder, device=device)
    model.apply(model_module.init_weights)

    epoch_loss = 0
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    for i, sample in enumerate(mnist_david.get_data(
                    batch_size=batch_size, n_classes=n_classes, n_rows=n_rows,
                    n_cols=n_cols,
                    train_size=train_size)):
        x, y = sample
        y = y.flatten()
        # x shape: (batch_size, channels=1, n_rows * 32, n_cols * 32)
        # y shape: (n_rows * n_cols)
        q, predicted_chars = model.forward(x, y)
        rewards = reward.get_reward(predicted_chars, y)

        q = q[torch.arange(q.shape[0]), predicted_chars]
        q2 = torch.cat((q[1:], torch.zeros(size=(1,))))
        q2.detach_()
        target_q = rewards + params['gamma'] * q2

        loss = criterion(q, target_q)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        epoch_loss += loss.item()
        if i % 100 == 0:
            print('iteration: {}'.format(i))
            _, predicted_chars = model.forward(x, y, debug=False)
            print(loss.item())
            print('Accuracy: {}'.format(
                sklearn.metrics.accuracy_score(y, predicted_chars)))


if __name__ == '__main__':
    with open(os.path.join('rl_reading', 'parameters.yaml'), 'r') as f:
        params = dict(yaml.load(f))
        params = params['default']

    # cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='outfiles/test1', help='destination of results')
    args = parser.parse_args()
    params['output_path'] = args.path

    # create main output directory
    if not os.path.exists(params['output_path']):
        os.makedirs(params['output_path'])
    else:
        inpt = input('WARNING: Output directory already exists! ' +
                     'Continue training? [y/N] (default: y)')
        if inpt.capitalize() == 'N':
            sys.exit('Ok exiting')

    with open(os.path.join(params['output_path'], 'parameters.yaml'), 'w') as f:
        yaml.dump(params, f)

    try:
        main(params)
    except KeyboardInterrupt:
        sys.exit('KeyboardInterrupt')
    except Exception:
        import traceback
        ty, value, tb = sys.exc_info()
        traceback.print_exc()
        try:
            import ipdb
            ipdb.post_mortem(tb)
        except ImportError:
            import pdb
            pdb.post_mortem(tb)
