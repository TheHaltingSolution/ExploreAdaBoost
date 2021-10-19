import numpy as np
from numpy.random import RandomState
from scipy.stats import bernoulli
from collections import namedtuple
ModelSample = namedtuple('ModelSample', ['X_true', 'y_true', 'X_sample', 'y_sample'])


class TrueModel:
    def __init__(self, model_function, domain_mins, domain_maxs, num_points_for_dim):
        self.model_function = model_function
        self.domain_mins = domain_mins
        self.domain_maxs = domain_maxs
        self.domain, y_true = self.get_function_graph(num_points_for_dim)
        self.y_true_scaled = self.scale_center_function(y_true)

    def get_uniform_domain_sample(self, random_seed):
        y_sampled = [self.get_sample(y_val, random_seed + idx) for idx, y_val in enumerate(self.y_true_scaled)]
        return ModelSample(self.domain, self.y_true_scaled, self.domain, y_sampled)

    # y_val is real number between -1 and 1
    # returns real number 0 and 1
    def get_bernoulli_prob(self, y_val):
        return y_val/2 + .5

    # return binary sample -1 or 1
    def get_sample(self, y_val, random_seed):
        bernoulli_sample = bernoulli.rvs(self.get_bernoulli_prob(y_val), size=1, random_state=RandomState(random_seed))[0]
        # convert to -1 or 1
        sample = -1 if bernoulli_sample == 0 else 1
        return sample

    def get_function_graph(self, num_points_for_dim):
        domain = self.get_coordinates_for_domain(self.domain_mins, self.domain_maxs, num_points_for_dim)
        y = [self.model_function(x) for x in domain]

        return domain, y

    def scale_center_function(self, function):
        max_y = np.max(function)
        min_y = np.min(function)
        shift = (max_y + min_y)/2
        scale = 2.0/np.abs(max_y - min_y)
        scaled_y = [scale * (y_val - shift) for y_val in function]

        return scaled_y

    def get_coordinates_for_domain(self, feature_mins, feature_maxs, num_points_for_dim):
        num_dims = min(len(feature_mins), len(feature_maxs))
        coordinate_arrays = []
        for dim in range(num_dims):
            coordinate_array = np.linspace(feature_mins[dim], feature_maxs[dim], num_points_for_dim[dim])
            coordinate_arrays.append(coordinate_array)
        feature_space_grids = np.meshgrid(coordinate_arrays)
        coordinate_tuples = np.array([np.ravel(xi) for xi in feature_space_grids]).reshape(-1, num_dims)
        return coordinate_tuples

    # gets cost of prediction function to true model
    # uses 0-1 cost function
    def get_empirical_risk(self, prediction_function):
        costs = []
        for y_true_val, y_predict in zip(self.y_true_scaled, prediction_function):
            costs.append(self.get_0_1_cost(y_true_val, y_predict))
        return sum(costs)/len(prediction_function)

    def get_0_1_cost(self, y_true_val, y_predict):
        y_prob = self.get_bernoulli_prob(y_true_val)
        if y_predict >= 0:
            return 1 - y_prob
        else:
            return y_prob

    def get_training_set_error(self, y_training, prediction_function):
        training_errors = 0.0
        training_set_size = float(len(y_training))

        for y_train, y_predict in zip(y_training, prediction_function):
            if (y_train == 1 and y_predict < 0) or (y_train == -1 and y_predict >= 0):
                training_errors = training_errors + 1
        return training_errors/training_set_size
