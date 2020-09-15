import numpy as np  


def normalize(data):
    """Return MinMax scaled data.
    
    Parameters
    ----------
    data : numpy.ndarray
        The data to be normalized.
    
    Returns
    -------
    numpy.ndarray
        Normalized data.
    """
    if isinstance(data, (np.ndarray)):
        assert data.ndim == 1, 'array must be 1-D'
    elif isinstance(data, (list)):
        data = np.array(data)
        assert data.ndim == 1, 'array must be 1-D'
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def restore(normalized, original):
    """Return restored data considering original data.
    
    Parameters
    ----------
    normalized : numpy.ndarray
        Scaled data.
    original : numpy.ndarray
        Original data.
    
    Returns
    -------
    numpy.ndarray
        Restored data.
    """
    if (isinstance(normalized, (np.ndarray)) and 
            isinstance(original, (np.ndarray))):
        assert normalized.ndim == 1 and original.ndim == 1, \
            'both arrays must be 1-D'
    elif isinstance(normalized, (list)) or isinstance(original, (list)):
        normalized = np.array(normalized)
        original = np.array(original)
        assert normalized.ndim == 1 and original.ndim == 1, \
            'both arrays must be 1-D'
    return normalized \
           * (np.max(original) - np.min(original)) \
           + np.min(original)


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
    assert y.ndim == 1, 'Data should be in array_like format.'
    if not isinstance(n, (int)):
        raise ValueError('Averaging windows should be integer.')
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
    if split_ratio < 0. or split_ratio > 1.:
        raise ValueError('split_ratio ill-defined.')
    train_size = int(round(split_ratio * len(data)))
    return data[:train_size], data[train_size:]