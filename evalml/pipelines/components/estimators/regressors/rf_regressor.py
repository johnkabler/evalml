from sklearn.base import clone as sk_clone
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class RandomForestRegressor(Estimator):
    """Random Forest Regressor"""
    name = "Random Forest Regressor"
    hyperparameter_ranges = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
    }
    model_family = ModelFamily.RANDOM_FOREST
    supported_problem_types = [ProblemTypes.REGRESSION]

    def __init__(self, n_estimators=100, max_depth=6, n_jobs=-1, random_state=0):
        parameters = {"n_estimators": n_estimators,
                      "max_depth": max_depth}
        rf_regressor = SKRandomForestRegressor(random_state=random_state,
                                               n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               n_jobs=n_jobs)
        super().__init__(parameters=parameters,
                         component_obj=rf_regressor,
                         random_state=random_state)

    @property
    def feature_importances(self):
        return self._component_obj.feature_importances_

    def clone(self):
        cloned_obj = RandomForestRegressor(n_estimators=self.parameters['n_estimators'],
                                           max_depth=self.parameters['max_depth'],
                                           random_state=self.random_state)
        cloned_regressor = sk_clone(self._component_obj)
        cloned_obj.rf_regressor = cloned_regressor
        return cloned_obj
