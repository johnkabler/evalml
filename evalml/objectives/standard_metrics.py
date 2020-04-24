import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from .binary_classification_objective import BinaryClassificationObjective
from .multiclass_classification_objective import MultiClassificationObjective
from .regression_objective import RegressionObjective

from evalml.exceptions import DimensionMismatchError


class AccuracyBinary(BinaryClassificationObjective):
    """Accuracy score for binary classification"""
    name = "Accuracy Binary"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        if len(y_true) == 0 or len(y_predicted) == 0:
            raise ValueError("Length of inputs is 0")
        if len(y_predicted) != len(y_true):
            raise DimensionMismatchError("Inputs have mismatched dimensions: y_predicted has shape {}, y_true has shape {}".format(len(y_predicted), len(y_true)))
        return metrics.accuracy_score(y_true, y_predicted)


class AccuracyMulticlass(MultiClassificationObjective):
    """Accuracy score for multiclass classification"""
    name = "Accuracy Multiclass"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        if len(y_true) == 0 or len(y_predicted) == 0:
            raise ValueError("Length of inputs is 0")
        if len(y_predicted) != len(y_true):
            raise DimensionMismatchError("Inputs have mismatched dimensions: y_predicted has shape {}, y_true has shape {}".format(len(y_predicted), len(y_true)))
        return metrics.accuracy_score(y_true, y_predicted)


class BalancedAccuracyBinary(BinaryClassificationObjective):
    """Balanced accuracy score for binary classification"""
    name = "Balanced Accuracy Binary"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.balanced_accuracy_score(y_true, y_predicted)


class BalancedAccuracyMulticlass(MultiClassificationObjective):
    """Balanced accuracy score for multiclass classification"""
    name = "Balanced Accuracy Multiclass"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.balanced_accuracy_score(y_true, y_predicted)


class F1(BinaryClassificationObjective):
    """F1 score for binary classification"""
    name = "F1"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.f1_score(y_true, y_predicted, zero_division=0.0)


class F1Micro(MultiClassificationObjective):
    """F1 score for multiclass classification using micro averaging"""
    name = "F1 Micro"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.f1_score(y_true, y_predicted, average='micro', zero_division=0.0)


class F1Macro(MultiClassificationObjective):
    """F1 score for multiclass classification using macro averaging"""
    name = "F1 Macro"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.f1_score(y_true, y_predicted, average='macro', zero_division=0.0)


class F1Weighted(MultiClassificationObjective):
    """F1 score for multiclass classification using weighted averaging"""
    name = "F1 Weighted"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.f1_score(y_true, y_predicted, average='weighted', zero_division=0.0)


class Precision(BinaryClassificationObjective):
    """Precision score for binary classification"""
    name = "Precision"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.precision_score(y_true, y_predicted, zero_division=0.0)


class PrecisionMicro(MultiClassificationObjective):
    """Precision score for multiclass classification using micro averaging"""
    name = "Precision Micro"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.precision_score(y_true, y_predicted, average='micro', zero_division=0.0)


class PrecisionMacro(MultiClassificationObjective):
    """Precision score for multiclass classification using macro averaging"""
    name = "Precision Macro"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.precision_score(y_true, y_predicted, average='macro', zero_division=0.0)


class PrecisionWeighted(MultiClassificationObjective):
    """Precision score for multiclass classification using weighted averaging"""
    name = "Precision Weighted"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.precision_score(y_true, y_predicted, average='weighted', zero_division=0.0)


class Recall(BinaryClassificationObjective):
    """Recall score for binary classification"""
    name = "Recall"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.recall_score(y_true, y_predicted, zero_division=0.0)


class RecallMicro(MultiClassificationObjective):
    """Recall score for multiclass classification using micro averaging"""
    name = "Recall Micro"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.recall_score(y_true, y_predicted, average='micro', zero_division=0.0)


class RecallMacro(MultiClassificationObjective):
    """Recall score for multiclass classification using macro averaging"""
    name = "Recall Macro"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.recall_score(y_true, y_predicted, average='macro', zero_division=0.0)


class RecallWeighted(MultiClassificationObjective):
    """Recall score for multiclass classification using weighted averaging"""
    name = "Recall Weighted"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.recall_score(y_true, y_predicted, average='weighted', zero_division=0.0)


class AUC(BinaryClassificationObjective):
    """AUC score for binary classification"""
    name = "AUC"
    greater_is_better = True
    score_needs_proba = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.roc_auc_score(y_true, y_predicted)


class AUCMicro(MultiClassificationObjective):
    """AUC score for multiclass classification using micro averaging"""
    name = "AUC Micro"
    greater_is_better = True
    score_needs_proba = True

    def objective_function(self, y_true, y_predicted, X=None):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='micro')


class AUCMacro(MultiClassificationObjective):
    """AUC score for multiclass classification using macro averaging"""
    name = "AUC Macro"
    greater_is_better = True
    score_needs_proba = True

    def objective_function(self, y_true, y_predicted, X=None):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='macro')


class AUCWeighted(MultiClassificationObjective):
    """AUC Score for multiclass classification using weighted averaging"""
    name = "AUC Weighted"
    greater_is_better = True
    score_needs_proba = True

    def objective_function(self, y_true, y_predicted, X=None):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='weighted')


class LogLossBinary(BinaryClassificationObjective):
    """Log Loss for binary classification"""
    name = "Log Loss Binary"
    greater_is_better = False
    score_needs_proba = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.log_loss(y_true, y_predicted)


class LogLossMulticlass(MultiClassificationObjective):
    """Log Loss for multiclass classification"""
    name = "Log Loss Multiclass"
    greater_is_better = False
    score_needs_proba = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.log_loss(y_true, y_predicted)


class MCCBinary(BinaryClassificationObjective):
    """Matthews correlation coefficient for binary classification"""
    name = "MCC Binary"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.matthews_corrcoef(y_true, y_predicted)


class MCCMulticlass(MultiClassificationObjective):
    """Matthews correlation coefficient for multiclass classification"""
    name = "MCC Multiclass"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.matthews_corrcoef(y_true, y_predicted)


class R2(RegressionObjective):
    """Coefficient of determination for regression"""
    name = "R2"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.r2_score(y_true, y_predicted)


class MAE(RegressionObjective):
    """Mean absolute error for regression"""
    name = "MAE"
    greater_is_better = False
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.mean_absolute_error(y_true, y_predicted)


class MSE(RegressionObjective):
    """Mean squared error for regression"""
    name = "MSE"
    greater_is_better = False
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.mean_squared_error(y_true, y_predicted)


class MedianAE(RegressionObjective):
    """Median absolute error for regression"""
    name = "MedianAE"
    greater_is_better = False
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.median_absolute_error(y_true, y_predicted)


class MaxError(RegressionObjective):
    """Maximum residual error for regression"""
    name = "MaxError"
    greater_is_better = False
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.max_error(y_true, y_predicted)


class ExpVariance(RegressionObjective):
    """Explained variance score for regression"""
    name = "ExpVariance"
    greater_is_better = True
    score_needs_proba = False

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.explained_variance_score(y_true, y_predicted)


def _handle_predictions(y_true, y_pred):
    if len(np.unique(y_true)) > 2:
        classes = np.unique(y_true)
        y_true = label_binarize(y_true, classes=classes)

    return y_true, y_pred
