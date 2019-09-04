import errno
import os
import shutil

import pandas as pd
import pytest

import evalml.tests as tests
from evalml import load_pipeline, save_pipeline
from evalml.objectives import Precision
from evalml.pipelines import LogisticRegressionPipeline
from evalml.pipelines.utils import get_pipelines, list_model_types
from evalml.problem_types import ProblemTypes

CACHE = os.path.join(os.path.dirname(tests.__file__), '.cache')


def test_list_model_types():
    assert set(list_model_types(ProblemTypes.BINARY)) == set(["random_forest", "xgboost", "linear_model"])
    assert set(list_model_types(ProblemTypes.REGRESSION)) == set(["random_forest"])


def test_get_pipelines():
    assert len(get_pipelines(problem_types=[ProblemTypes.BINARY])) == 3
    assert len(get_pipelines(problem_types=[ProblemTypes.BINARY], model_types=["linear_model"])) == 1
    assert len(get_pipelines(problem_types=[ProblemTypes.REGRESSION])) == 1


@pytest.fixture
def path_management():
    path = CACHE
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:  # EEXIST corresponds to FileExistsError
            raise e
    yield path
    shutil.rmtree(path)


def test_serialization(X_y, trained_model, path_management):
    X, y = X_y
    X = pd.DataFrame(X)
    y = pd.Series(y)
    path = os.path.join(path_management, 'pipe.pkl')
    objective = Precision()

    pipeline = LogisticRegressionPipeline(objective=objective, penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]))
    pipeline.fit(X, y)
    save_pipeline(pipeline, path)
    assert pipeline.score(X, y) == load_pipeline(path).score(X, y)
