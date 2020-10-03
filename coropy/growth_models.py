import warnings

import numpy as np 
from scipy.optimize import curve_fit 

from .utils import normalize, restore


__all__ = ['GrowthCOVIDModel', '_exp_func', '_logistic_func']


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


_growth_model_dispatcher = {
    'exponential': _exp_func,
    'logistic': _logistic_func,
    }


class GrowthCOVIDModel(object):
    """A class to fit an exponential model to given data."""
    def __init__(self, function, normalize=True, calc_ci=False, **kwargs):
        """Constructor.
        
        Parameters
        ----------
        function : str
            Growth curve.
        normalize : bool, optional
            Data is normalized to [0, 1] range.
        calc_ci : bool, optional
            Generate CI using fitted params of a given function +/-
            standard deviation.
        **kargs : dict, optional
            Additional keyword argument for `curve_fit` function.
        """
        try:
            self.function = _growth_model_dispatcher[function]
        except:
            raise ValueError('Try `exponential` or `logistic` growth models.')
        self.normalize = normalize
        self.calc_ci = calc_ci
        if kwargs:
            self.acc = kwargs['acc']
        else:
            self.acc = None
    
    def fit(self, data):
        """Fit the data to the growth function.
        
        Parameters
        ----------
        data : numpy.ndarray
            Cumulative number of cumulative infectious cases.
        
        Returns
        -------
        x : numpy.ndarray
            The independent variable where the data is measured.
        fitted : numpy.ndarray
            Fitted growth function with lower and upper bound if
            `confidence_interval` is set to True.
        """
        self.data = data
        if self.normalize:
            y = normalize(self.data)
            x = normalize(np.arange(self.data.size))
        else: 
            y = self.data
            x = np.arange(self.data.size)

        if self.calc_ci and self.acc:
            y_std = y*self.acc
            abs_std = True
        else:
            y_std = np.ones_like(y)
            abs_std = False

        self.popt, self.pcov = curve_fit(self.function, x, y, sigma=y_std,
            absolute_sigma=abs_std)
        fitted = self.function(x, *self.popt)

        if self.calc_ci:
            self.perr = np.sqrt(np.diag(self.pcov))
            lower_bound = self.function(x, *(self.popt - self.perr))
            upper_bound = self.function(x, *(self.popt + self.perr))
            if self.normalize: 
                x = restore(x, np.arange(self.data.size))
                fitted = restore(fitted, self.data)
                lower_bound = restore(lower_bound, self.data)
                upper_bound = restore(upper_bound, self.data)
            fitted = np.r_[
                lower_bound.reshape(1, -1), 
                fitted.reshape(1, -1), 
                upper_bound.reshape(1, -1)]

        elif not self.calc_ci and self.normalize:
            x = restore(x, np.arange(self.data.size))
            fitted = restore(fitted, self.data)
        return x, fitted

    def predict(self, n_days):
        """Predict the future n_days using the fitted growth function.
        
        Parameters
        ----------
        n_days : int
            Number of days to extrapolate.
        
        Returns
        -------
        x_fut : numpy.ndarray
            The independent variable where the data is predicted.
        predicted : numpy.ndarray
            Extrapolated data with lower and upper bound if 
            `confidence_interval` is set to True.
        """
        assert isinstance(n_days, (int,)), 'Number of days must be integer.'
        if self.normalize:
            x = normalize(np.arange(self.data.size))
            delta_x = np.diff(x[:2])[0]
            fut_start_norm = np.max(x) + delta_x
            fut_end_norm = fut_start_norm + (n_days - 1)*delta_x
            x_fut = np.linspace(fut_start_norm, fut_end_norm, n_days)
        else: 
            x = np.arange(self.data.size)
            x_fut = np.arange(self.data.size, self.data.size + n_days)
        
        predicted = self.function(x_fut, *self.popt)

        if self.calc_ci:
            lower_bound = self.function(x_fut, *(self.popt - self.perr))
            upper_bound = self.function(x_fut, *(self.popt + self.perr))
            if self.normalize: 
                x_fut = restore(x_fut, np.arange(self.data.size))
                predicted = restore(predicted, self.data)
                lower_bound = restore(lower_bound, self.data)
                upper_bound = restore(upper_bound, self.data)
            predicted = np.r_[
                lower_bound.reshape(1, -1), 
                predicted.reshape(1, -1), 
                upper_bound.reshape(1, -1)]
        elif not self.calc_ci and self.normalize:
            x_fut = restore(x_fut, np.arange(self.data.size))
            predicted = restore(predicted, self.data)
        return x_fut, predicted