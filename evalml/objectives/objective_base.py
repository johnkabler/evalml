from abc import ABC, abstractmethod


class ObjectiveBase(ABC):
    name = None
    greater_is_better = True
    score_needs_proba = False

    def __init__(self):
        if self.name is None:
            raise NameError("Objective `name` cannot be set to None.")

    @abstractmethod
    def objective_function(self, y_predicted, y_true, X=None):
        raise NotImplementedError

    # def score(self, y_predicted, y_true, X=None):
    #     """Calculate score from applying fitted objective to predicted values

    #     Arguments:
    #         y_predicted (list): the predictions from the model. If needs_proba is True,
    #             it is the probability estimates

    #         y_true (list): the ground truth for the predictions.

    #         X (pd.DataFrame): any extra columns that are needed from training
    #             data to fit. Only provided if uses_extra_columns is True.

    #     Returns:
    #         score

    #     """
    #     return self.objective_function(y_predicted, y_true, X)
