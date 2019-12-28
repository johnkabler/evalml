import pandas as pd

from evalml.utils import Logger


def summarize_table(X):
    """
    Arguments:
        X (pd.DataFrame)
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    logger = Logger()
    logger.log_title("Summary for table:")

    logger.log("Number of rows: {}".format(X.shape[0]))
    logger.log("Number of columns: {}".format(X.shape[1]))

    logger.log("Number of columns by data type:")
    counts = X.dtypes.value_counts()
    for dtype in counts.index: 
        logger.log('{}: {}'.format(dtype, counts[dtype]))
    X.dtypes.value_counts()

    logger.log("Total size of DataFrame: {} bytes".format(X.memory_usage(index=True).sum()))



def summarize_row(X, row):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    logger = Logger()
    logger.log("Number of non-NaN elements in row {}: {}".format(row, X.count(axis=1)[row]))


def summarize_col(X, col):
    # todo: combine with summarize_row but change axis?
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    logger = Logger()
    logger.log("Number of non-NaN elements in col {}: {}".format(col, X.count(axis=0)[col]))
