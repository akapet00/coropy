from collections import Iterable

import numpy as np
from sklearn.preprocessing import MinMaxScaler as Scaler


def normalize(data):
    """Return MinMax scaled data.
    
    Parameters
    ----------
    data : iterable
        The data to be normalized.
    
    Returns
    -------
    numpy.ndarray
        Normalized data.
    """
    if not isinstance(data, Iterable):
        raise ValueError('data must be iterable')
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    if isinstance(data, (np.ndarray)):
        assert data.ndim == 1, 'data must be 1-D'
        assert data.size > 1, 'data must contain at least 2 elements'
    if np.isnan(data).any():
        raise ValueError('data contains one or more NaN values')
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def restore(y_norm, y):
    """Return restored data considering original data.
    
    Parameters
    ----------
    y_norm : iterable
        Scaled data.
    y : iterable
        Original data.
    
    Returns
    -------
    numpy.ndarray
        Restored data.
    """
    if not (isinstance(y_norm, Iterable) and
            isinstance(y, Iterable)):
        raise ValueError('data must be iterable')
    if isinstance(y_norm, (list, tuple)):
        y_norm = np.array(y_norm)
    if isinstance(y, (list, tuple)):
        y = np.array(y)
    if (isinstance(y_norm, (np.ndarray)) and 
            isinstance(y, (np.ndarray))):
        assert y_norm.ndim == 1 and y.ndim == 1, \
            'data must be 1-D'
        assert y_norm.size > 1 or y.size > 1, \
            'data must contain at least 2 elements'
    if np.isnan(y_norm).any() or np.isnan(y).any():
        raise ValueError('data contains one or more NaN values')
    return y_norm * (np.max(y) - np.min(y)) + np.min(y)


def moving_average(y, n=3):
    """Return a array of averages of different subsets of the full data
    set.

    Parameters
    ----------
    y : numpy.ndarray
        Data to be averaged.
    n : int
        Averaging window.

    Returns
    -------
    numpy.ndarray
        Averaged data.
    """
    if not isinstance(n, (int)):
        raise ValueError('Averaging windows should be an integer.')
    if not isinstance(y, Iterable):
        raise ValueError('data must be iterable')
    if isinstance(y, (list, tuple)):
        y = np.array(y)
    if isinstance(y, (np.ndarray)):
        assert y.ndim == 1, 'data must be 1-D'
        assert y.size >= n, 'data must contain at least `n` elements'
    if n == 0:
        return y
    ret = np.cumsum(y, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def mse(y_true, y_pred):
    """Return mean square difference between two values.
    
    Parameters
    ----------
    y_true : float or numpy.ndarray
        True value(s).
    y_pred : float or numpy.ndarray
        Predicted or simulated value(s).
    
    Returns
    -------
    float
        Mean square error value.
    """
    if (not isinstance(y_true, (np.ndarray, int, float)) or 
            not isinstance(y_pred, (np.ndarray, int, float))):
        raise ValueError('supported data types: numpy.ndarray, int, float')
    return np.mean((y_true - y_pred)**2)


def rmse(y_true, y_pred):
    """Return root mean square difference between two values.
    
    Parameters
    ----------
    y_true : float or numpy.ndarray
        True value(s).
    y_pred : float or numpy.ndarray
        Predicted or simulated value(s).
    
    Returns
    -------
    float
        Root mean square error value.
    """
    if (not isinstance(y_true, (np.ndarray, int, float)) or 
            not isinstance(y_pred, (np.ndarray, int, float))):
        raise ValueError('supported data types: numpy.ndarray, int, float')
    return np.sqrt(mse(y_true, y_pred))


def msle(y_true, y_pred):
    """Return mean square log difference between two values.
    
    Parameters
    ----------
    y_true : float or numpy.ndarray
        True value(s).
    y_pred : float or numpy.ndarray
        Predicted or simulated value(s).
    
    Returns
    -------
    float
        Mean square log error value.
    """
    if (not isinstance(y_true, (np.ndarray, int, float)) or 
            not isinstance(y_pred, (np.ndarray, int, float))):
        raise ValueError('supported data types: numpy.ndarray, int, float')
    return mse(np.log1p(y_true), np.log1p(y_pred))


def mae(y_true, y_pred):
    """Return mean absolute difference between two values.
    
    Parameters
    ----------
    y_true : float or numpy.ndarray
        True value(s).
    y_pred : float or numpy.ndarray
        Predicted or simulated value(s).
    
    Returns
    -------
    float
        Mean absolute error value.
    """
    if (not isinstance(y_true, (np.ndarray, int, float)) or 
            not isinstance(y_pred, (np.ndarray, int, float))):
        raise ValueError('supported data types: numpy.ndarray, int, float')
    return np.mean(np.abs(y_true - y_pred))


def train_test_split(data, split_ratio=0.8):
    """Returns data split into two parts, train and test part.
    
    Parameters
    ----------
    data : numpy.ndarray
        Full single dimensional data set.
    split_ratio : float, optional
        Ratio for data split.
        
    Returns
    -------
    numpy.array and numpy.array
        Train data set and test data set.
    """
    if not isinstance(data, (np.ndarray,)):
        raise ValueError('data must be numpy.ndarray')
    if split_ratio < 0. or split_ratio > 1.:
        raise ValueError('`split_ratio` ill-defined, must be in <0, 1> range')
    train_size = int(round(split_ratio * len(data)))
    return data[:train_size], data[train_size:]