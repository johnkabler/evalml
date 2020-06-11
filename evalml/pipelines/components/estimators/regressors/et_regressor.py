from sklearn.ensemble import ExtraTreesRegressor as SKExtraTreesRegressor
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class ExtraTreesRegressor(Estimator):
    """Extra Trees Regressor"""
    name = "Extra Trees Regressor"
    hyperparameter_ranges = {
        "n_estimators": Integer(10, 1000),
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": Integer(4, 10)
    }
    model_family = ModelFamily.EXTRA_TREES
    supported_problem_types = [ProblemTypes.REGRESSION]

    def __init__(self,
                 n_estimators=100,
                 max_features="auto",
                 max_depth=6,
                 min_samples_split=2,
                 min_weight_fraction_leaf=0.0,
                 n_jobs=-1,
                 random_state=0):
        parameters = {"n_estimators": n_estimators,
                      "max_features": max_features,
                      "max_depth": max_depth,
                      "min_samples_split": min_samples_split,
                      "min_weight_fraction_leaf": min_weight_fraction_leaf}
        et_regressor = SKExtraTreesRegressor(random_state=random_state,
                                             n_estimators=parameters['n_estimators'],
                                             max_features=parameters['max_features'],
                                             max_depth=parameters['max_depth'],
                                             min_samples_split=parameters['min_samples_split'],
                                             min_weight_fraction_leaf=parameters['min_weight_fraction_leaf'],
                                             n_jobs=n_jobs)
        super().__init__(parameters=parameters,
                         component_obj=et_regressor,
                         random_state=random_state)
