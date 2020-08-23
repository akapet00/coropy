import numpy as np 
from scipy.optimize import curve_fit 

from covid_19.utils import normalize, restore


def _exp_func(x, a, b, c):
    """Exponential function of a single variable, x.
    
    Parameters
    ----------
    x : float or numpy.ndarray
        Input data.
    a : float
        First parameter.
    b : float
        Second parameter.
    c : float
        Third parameter.
    
    Returns
    -------
    float or numpy.ndarray
        a * exp(b * x) + c
    """
    return a * np.exp(b * x) + c


def _logistic_func(x, a, b, c, d):
    """Logistic function of a single variable, x.
    
    Parameters
    ----------
    x : float or numpy.ndarray
        Input data.
    a : float
        First parameter.
    b : float
        Second parameter.
    c : float
        Third parameter.
    d : float
        Forth parameter.
    
    Returns
    -------
    float or numpy.ndarray
        a/(1 + exp(-c *(x-d)))+b
    """
    return a / (1 + np.exp(-c * (x - d))) + b


_dispatcher = {
    'exponential': _exp_func,
    'logistic': _logistic_func,
    }


class GrowthCOVIDModel(object):
    """A class to fit an exponential model to given data."""  
    def __init__(self, function, normalize=True):
        """Constructor.
        
        Parameters
        ----------
        function : str
            Growth curve.
        normalize : bool, optional
            Should the data be normalized to [0, 1] range.
        """
        try:
            self.function = _dispatcher[function]
        except:
            raise ValueError('Try `exponential` or `logistic` growth models.')
        self.normalize = normalize
        
    def fit(self, confirmed_cases):
        """Fit the data to the growth function.
        
        Parameters
        ----------
        confirmed_cases : numpy.ndarray
            Number of confirmed infected COVID-19 cases per day.
        
        Returns
        -------
        x : numpy.ndarray
            The independent variable where the data is measured.
        fitted : numpy.ndarray
            Fitted growth function.
        """
        self.confirmed_cases = confirmed_cases

        if self.normalize:
            y = normalize(confirmed_cases)
        else: 
            y = confirmed_cases
    
        x = np.arange(confirmed_cases.size)
        self.popt, self.pcov = curve_fit(self.function, x, y)
        fitted = self.function(x, *self.popt) 
        if self.normalize: 
            fitted = restore(fitted, confirmed_cases)
        return x, fitted

    def predict(self, n_days):
        """Predict the future n_days using the fitted growth function.
        
        Parameters
        ----------
        n_days : int
            Number of days to extrapolate.
        
        Returns
        -------
        x_future : numpy.ndarray
            The independent variable where the data is predicted.
        predicted : numpy.ndarray
            Extrapolated data.
        """
        assert isinstance(n_days, (int,)), 'Number of days must be integer.'
        size = self.confirmed_cases.size
        x_future = np.arange(size-1, size+n_days)
        predicted = self.function(x_future, *self.popt)
        if self.normalize: 
            predicted = restore(predicted, self.confirmed_cases)
        return x_future, predicted