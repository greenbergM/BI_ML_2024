import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    int_y_pred = y_pred.astype(int)
    int_y_true = y_true.astype(int)

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for idx in range(y_true.shape[0]):
        if int_y_true[idx] == 1 and int_y_pred[idx] == 1:
            true_positive += 1
        if int_y_true[idx] == 0 and int_y_pred[idx] == 0:
            true_negative += 1
        if int_y_true[idx] == 0 and int_y_pred[idx] == 1:
            false_positive += 1
        if int_y_true[idx] == 1 and int_y_pred[idx] == 0:
            false_negative += 1

    if (true_positive + true_negative + false_positive + false_negative) != 0:
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    else:
        accuracy = None

    if (true_positive + false_positive) != 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = None

    if (true_positive + false_negative) != 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = None

    if (precision or recall) is not None:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    positive_results = 0

    for idx in range(y_true.shape[0]):
        if y_pred[idx] == y_true[idx]:
            positive_results += 1

    accuracy = positive_results/y_true.shape[0]
    return accuracy


def r_squared_metric(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return r2


def mse_metric(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = np.sum((y_true - y_pred) ** 2)/len(y_true)
    return mse


def mae_metric(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = np.sum(abs(y_true - y_pred))/len(y_true)
    return mae