import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckWarning

from evalml.utils.gen_utils import numeric_and_boolean_dtypes
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

class LabelLeakageDataCheck(DataCheck):
    """Check if any of the features are highly correlated with the target."""

    def __init__(self, pct_corr_threshold=0.95, method='correlation', problem_type=None):
        """Check if any of the features are highly correlated with the target.

        Currently only supports binary and numeric targets and features.

        Arguments:
            pct_corr_threshold (float): The correlation threshold to be considered leakage. 
                Defaults to 0.95. Only used for `correlation`. Maybe use this for both?
            method (string): The method used to determine label leakage. Currently supports `correlation` and
                `mutual_info`. Defaults to `correlation`.
            problem_type (ProblemTypes enum): Only used if method is 'mutual_info`.
        """
        if pct_corr_threshold < 0 or pct_corr_threshold > 1:
            raise ValueError("pct_corr_threshold must be a float between 0 and 1, inclusive.")
        self.pct_corr_threshold = pct_corr_threshold
        self.method = method
        self.problem_type = problem_type
  
    def validate(self, X, y):
        """Check if any of the features are highly correlated with the target.

        Currently only supports binary and numeric targets and features.

        Arguments:
            X (pd.DataFrame): The input features to check
            y (pd.Series): The labels

        Returns:
            list (DataCheckWarning): list with a DataCheckWarning if there is label leakage detected.

        Example:
            >>> X = pd.DataFrame({
            ...    'leak': [10, 42, 31, 51, 61],
            ...    'x': [42, 54, 12, 64, 12],
            ...    'y': [12, 5, 13, 74, 24],
            ... })
            >>> y = pd.Series([10, 42, 31, 51, 40])
            >>> label_leakage_check = LabelLeakageDataCheck(pct_corr_threshold=0.8)
            >>> assert label_leakage_check.validate(X, y) == [DataCheckWarning("Column 'leak' is 80.0% or more correlated with the target", "LabelLeakageDataCheck")]
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if y.dtype not in numeric_and_boolean_dtypes:
            return []
        X = X.select_dtypes(include=numeric_and_boolean_dtypes)
        if len(X.columns) == 0:
            return []

        if self.method == 'correlation':
            highly_corr_cols = {label: abs(y.corr(col)) for label, col in X.iteritems() if abs(y.corr(col)) >= self.pct_corr_threshold}
            warning_msg = "Column '{}' is {}% or more correlated with the target"
        elif self.method == 'mutual_info':
            mutual_info = mutual_info_classif(X, y)
            import pdb; pdb.set_trace()
            highly_corr_cols = {label: mutual_info[col] for label, col in X.iteritems() if abs(mutual_info[col]) >= self.pct_corr_threshold}
        return [DataCheckWarning(warning_msg.format(col_name, self.pct_corr_threshold * 100), self.name) for col_name in highly_corr_cols]
