from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
import numpy as np


from modelgenerator import TrueModel, ModelSample
from exploregui import ExploreGui
from collections import namedtuple
AxisRange = namedtuple('AxisRange', ['min', 'max'])
ModelOutputs = namedtuple('ModelOutputs', ['sample', 'axis_range', 'prediction_functions', 'true_error',
                                           'training_error', 'generalization_error'])

def main():
    # models
    model_sq = lambda argument_array: argument_array[0] ** 2
    model_sin = lambda argument_array: np.sin(argument_array[0])
    model_sin3x = lambda argument_array: np.sin(3 * argument_array[0])
    model_sinxd3 = lambda argument_array: np.sin(argument_array[0]/3.0)
    model = model_sinxd3

    domain_min = [-10]
    domain_max = [10]
    points_per_dimension = [200]

    experiment_model = TrueModel(model, domain_min, domain_max, points_per_dimension)
    model_sample = experiment_model.get_uniform_domain_sample(123)

    clf = AdaBoostClassifier(n_estimators=100, random_state=0, algorithm='SAMME')
    clf.fit(model_sample.X_sample, model_sample.y_sample)

    estimator_list = clf.estimators_
    estimator_weights = clf.estimator_weights_
    cumulative_prediction_functions = get_cum_prediction_functions(estimator_list, estimator_weights, model_sample.X_true)
    axis_range = get_y_bounds(cumulative_prediction_functions, .10)

    training_error = clf.estimator_errors_
    training_error_cum = [experiment_model.get_training_set_error(model_sample.y_sample, prediction_function) for prediction_function in cumulative_prediction_functions]
    true_error = [experiment_model.get_empirical_risk(prediction_function) for prediction_function in cumulative_prediction_functions]
    generalization_error = np.subtract(true_error, training_error_cum)

    model_outputs = ModelOutputs(model_sample, axis_range, cumulative_prediction_functions, true_error, training_error_cum, generalization_error)

    gui = ExploreGui()
    gui.setup_ui(model_outputs)
    gui.start_ui()

    return


def get_training_set_1dim(feature_min=-10, feature_max=10, num_instances=20):
    X = np.linspace(feature_min, feature_max, num=num_instances).reshape(-1, 1)
    y_positive = np.asarray([1, -1, 1, 1, -1, 1, 1, 1, 1, 1])
    y = np.append(np.flip(y_positive) * -1, y_positive)
    return X, y


def get_cum_prediction_functions(estimator_list, estimator_weights, coordinate_tuples):
    domain_size = coordinate_tuples.size

    cumulative_prediction_functions = []
    cumulative_prediction_function = np.zeros(domain_size)

    for estimator, weight in zip(estimator_list, estimator_weights):
        y = estimator.predict(coordinate_tuples)
        cumulative_prediction_function = cumulative_prediction_function + y * weight
        cumulative_prediction_functions.append(cumulative_prediction_function)

    return cumulative_prediction_functions


def get_prediction_function(classifier, coordinate_tuples):
    y = classifier.decision_function(coordinate_tuples)

    return coordinate_tuples, y

# margin percentage to create some extra space
def get_y_bounds(prediction_functions, margin_percentage):
    y_maxs = [np.max(function) for function in prediction_functions]
    y_mins = [np.min(function) for function in prediction_functions]
    return AxisRange(np.min(y_mins) * (1 + margin_percentage), np.max(y_maxs) * (1 + margin_percentage))

def get_training_set_random():
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    return X, y


if __name__ == "__main__":
    main()
