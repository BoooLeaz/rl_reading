import numpy as np
import signal
import logging
import subprocess
import warnings

logger = logging.getLogger('general')


def radians(angle):
    return angle / 360 *  2 * np.pi


def project_into_plane(arr, angle):
    """project_into_plane

    :param arr: array-like, shape (..., 2), or (2,)
        contains samples in rows and x, y position in columns
    :param angle: scalar, angle with x-axis in which to project (radians)

    :Returns: array, shape like arr (input array)
    """
    orig_shape = arr.shape
    arr = arr.reshape((-1, 2))

    projection_vec = np.array([np.cos(angle), np.sin(angle)]).reshape((2, 1))
    projection = np.dot(arr, projection_vec) * projection_vec.reshape((1, 2))
    return projection.reshape(orig_shape)


def z_values(arr, axis=None):
    zvalues = arr - np.mean(arr, axis=axis, keepdims=True)
    zvalues /= (np.std(arr, axis=axis, keepdims=True) + 1e-6)
    return zvalues


def get_rupture_closeness(next_positions, ruptures, scaling=0.1):
    """get_rupture_closeness

    :param next_positions:  array, shape (..., 3) with x, y, z in the last dimension
    :param ruptures:  array, shape (..., 3) with x, y, z in the last dimension
    """
    next_positions_orig_shape = next_positions.shape
    next_positions = next_positions.reshape((-1, 3))
    ruptures = ruptures.reshape((-1, 3))

    # handle case where there are no ruptures
    if ruptures.shape[0] == 0:
        return np.zeros(shape=next_positions_orig_shape[:-1])

    rbf = RBFComputation(ruptures[:, 0], ruptures[:, 1], ruptures[:, 2], grid=False,
            scaling=scaling)
    rbf_distances = rbf.get_rbf_features(next_positions)  # shape (n_next_positions, n_ruptures)
    rupture_closeness = np.max(rbf_distances, axis=-1)
    rupture_closeness = rupture_closeness.reshape(next_positions_orig_shape[:-1])

#    avg_closest_n = 1  # assign closeness of closest n ruptures
#    if rbf.n_centers < avg_closest_n:
#        rupture_closeness = np.mean(rbf_distances, axis=1)
#    else:
#        # take the median of the top n distances to determine how close we are to a 'wall'
#        rupture_closeness = np.mean(
#            np.partition(
#                rbf_distances,
#                rbf.n_centers - avg_closest_n - 1, axis=1)[:, - avg_closest_n:], axis=1)
    return rupture_closeness


class RBFComputation:
    def __init__(self, x1, x2, x3, grid=True, scaling=4.0):
        """__init__

        :param x1: array (n_centers,)
        :param x2: array (n_centers,)
        :param x3: array (n_centers,)
        :param grid: bool
            if True, treat x1, x2, x3 as coordinates along which to span a grid of RBF centers
            if False, treat x1, x2, x3 as coordinates of RBF centers
        :param scaling: RBF scaling
        """
        self.scaling = scaling
        if grid:
            self.centers = np.stack(
                np.meshgrid(x1, x2, x3, copy=True), axis=-1).reshape((-1, 3), order='F')
        else:
            self.centers = np.stack((x1, x2, x3), axis=1)
        self.n_centers = self.centers.shape[0]

    def get_rbf_features(self, data, override_scaling=None):
        """get_rbf_features

        :param data: shape (..., 3)
        :param override_scaling: float or none
            If not None, override the scaling attribute with the given float

        Returns: rbf_features, shape (input_shape[:-1] + (n_rbf_centers,))
        """
        scaling = self.scaling if override_scaling is None else override_scaling

        input_shape = data.shape
        if not input_shape[-1] == 3:
            raise ValueError('last dimension of dta needs to be 3 (x, y, z)')

        # reshaping here to do numpy broadcasting
        data = data.reshape((-1, 1, 3))
        coordinate_distances = data - self.centers.reshape((1, self.n_centers, 3))
        # in the following we compute the nominator of the exp in the gaussian
        # since it is squared we can just avoid taking the square root of the 
        # euclidean norm
        squared_distances = np.sum(coordinate_distances**2, axis=-1)
        rbf_features = np.exp(- squared_distances / scaling)
        return rbf_features.reshape(input_shape[:-1] + (self.n_centers,))

    def update_centers(self, new_centers):
        self.centers = new_centers


class Ignore_KeyboardInterrupt:
    def __enter__(self):
        def handle_exception(signal_, frame):
            logger.warning('')
            logger.warning('You tried to interrupt the program. Currently this is not a good time.')
            logger.warning('Wait a few seconds, please.')
            logger.warning('Otherwise, interrupt again and live with the consequences.')
            # reinstate the original handler in case we get another interrupt
            signal.signal(signal.SIGINT, self.original_handler)
        self.original_handler = signal.signal(signal.SIGINT, handle_exception)

    def __exit__(self, exc_type, exc_value, exc_tb):
        signal.signal(signal.SIGINT, self.original_handler)


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip('\n')
    except Exception:
        warnings.warn('\nWarning! Failed to get git revision hash!\n')
        return('FAILED_to_get_git_revision_hash')


def extract_position(data):
    '''
    :param data: array-like, shape (batch, steps, 5)
        :3 position
        3 df
        4 current
    :return:
    '''
    if len(data.shape) == 1:  # we'll assume it's just one sample
        data = data.reshape((1, 1, -1))
    elif len(data.shape) != 3:
        raise Exception(
            'Data needs to have 3 dimensions. You passed {}'.format(len(data.shape)))

    features = data[:, :, :3]
    return features
