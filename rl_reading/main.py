import argparse
import logging
import os
import sys
import time
import shutil
import numpy as np
from ruamel.yaml import YAML
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# imports from this project
from reward import Reward
import agents
import models
import util
import mnist_david

yaml = YAML()
logger = logging.Logger('main_logger')


def main(params):
    start_time = time.time()
    output_path = params['output_path']
    if 'gridsearch' in params.keys():
        gridsearch_results_path = params['gridsearch_results_path']
        gridsearch = params['gridsearch']
    else:
        gridsearch = False

    plot_interval = 10 if gridsearch else 1
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    with open(os.path.join(output_path, 'parameters.yaml'), 'w') as f:
        yaml.dump(params, f)

    # create results directories
    model_dir = os.path.join(output_path, 'models')
    data_basepath = os.path.join(output_path, 'data')
    all_paths = [model_dir, data_basepath]
    for _path in all_paths:
        if not os.path.exists(_path):
            os.makedirs(_path)

    # initialize/load program state
    start_program_state = {
        'episode': -1,
        'global_step': 0,
        'best_episode_score': 0,
        'best_episode_index': 0,
        'output_path': output_path,
        'git_head': util.get_git_revision_hash(),
        'run_finished': False,
        'start_time': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(start_time))
    }
    program_state = start_program_state
    program_state_path = os.path.join(output_path, 'program_state.yaml')
#    try:
#        # TODO
#        with open(program_state_path, 'r') as f:
#            program_state = yaml.load(f)
#        logger.info('Loaded program_state')
#    except Exception:
#        logger.debug('Did not find program_state.dump file to restore program state, starting new')
#        program_state = start_program_state

    # ACTIONS
    characters = np.array(["<sos>", "<eos>", "0", "1", "linefeed"])
    actions    = np.array([                  "0", "1", "linefeed"])

    # setup models
    # Reward
#    reward = Reward(step_reward=params['step_reward'], fail=params['fail_reward'], success=1.0)
    # QModels
    model_module = getattr(models, params['model'])
    encoder = model_module.Encoder(params)
    decoder = model_module.Decoder(params, n_actions=len(actions), n_characters=len(characters))
    model = model_module.EncoderDecoder(encoder, decoder, device=device)
    model.apply(model_module.init_weights)
#    target_model = model_module.Model(params, actions)
#    current_model_path = os.path.join(output_path, 'models', 'current')
#    previous_model_path = os.path.join(output_path, 'models', 'previous')
#    try:
#        logger.info('Loading model from: ' + current_model_path)
#        model = torch.load(current_model_path)
#        target_model = torch.load(current_model_path)
#    except Exception:
#        model.save(current_model_path)
#        model.save(previous_model_path)

    n_rows = 1
    n_cols = 1
    batch_size = 1
    n_classes = 2
    train_size = 100
    for x, y in mnist_david.get_data(
                    batch_size=batch_size, n_classes=n_classes, n_rows=n_rows,
                    n_cols=n_cols,
                    train_size=train_size):
        outputs = model(x, y)
        import ipdb; ipdb.set_trace()

    # agent
#    agent_module = getattr(agents, params['agent'])
#    agent = agent_module.Agent(
#        params=params,
#        model=model,
#        target_model=target_model,
#        actions=actions,
#        reward=reward,
#        global_step=program_state['global_step'],
#        episode=program_state['episode'])

    logger.info('Starting at training step ' + str(agent.global_step))
    while True:

        agent.reset()
        agent.run_episode()

        # update program state
        program_state['episode'] = int(agent.episode)
        program_state['global_step'] = int(agent.global_step)

        # save everything to disk. That process should not be interrupted.
        with util.Ignore_KeyboardInterrupt():
            with open(program_state_path, 'w') as f:
                yaml.dump(program_state, f)
            # on disk, copy the current model file to previous model, then save the current
            # in-memory model to disk as the current model
            shutil.copy(current_model_path, previous_model_path)
            model.save(current_model_path)
            model.save(os.path.join(output_path, 'models', 'episode_{}'.format(agent.episode)))


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
#    else:
#        inpt = input('WARNING: Output directory already exists! ' +
#                     'Continue training? [y/N] (default: y)')
#        if inpt.capitalize() == 'N':
#            sys.exit('Ok exiting')

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
