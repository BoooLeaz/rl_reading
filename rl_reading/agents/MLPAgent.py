import numpy as np
np.set_printoptions(precision=3, floatmode='fixed', sign='+')
import logging
import torch
import matplotlib.pyplot as plt
import os
from scipy.special import softmax

import util
import optimizer_util
import plotting
from . import base_agent

logger = logging.getLogger('general')
action_logger = logging.getLogger('action')
trajectory_logger = logging.getLogger('trajectory')
# just to append the episode beginning/end strings:
measurement_logger = logging.getLogger('measurement')


# in the current version of the code, this agent is equivalent to the base agent
class Agent(base_agent.Agent):
    pass
