from abc import abstractmethod

from .objective_base import ObjectiveBase


class ClassificationObjective(ObjectiveBase):

    @abstractmethod
    def decision_function(self, ypred_proba, threshold=None, X=None):
        raise NotImplementedError("decision_function for this classification objective is not yet defined!")