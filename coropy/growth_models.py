import warnings

import numpy as np 
from scipy.optimize import curve_fit 

from .utils import Scaler, normalize, restore


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
        if isinstance(function, (str, )):
            try:
                self.function = _growth_model_dispatcher[function]
            except:
                raise ValueError('Try `exponential` or `logistic` growth models.')
        elif callable(function):
            self.function = function
        else:
            raise TypeError('Argument type for `function` is inappropriate.')
        self.normalize = normalize
        self.calc_ci = calc_ci
        if kwargs:
            try:
                self.spread = kwargs['spread']
            except KeyError:
                warnings.warn(
                    'Keyword argument not understood.'
                    'Confidence intervals will be plotted without `spread`.')
                self.spread = None
            finally:
                pass
        else:
            self.spread = None
    
    def __str__(self):
        return(
            f'Epidemic growth model \n',
            f'--------------------- \n',
            f'- {self.function} function assumption \n'
            f'- normalization of the data: {self.normalize} \n'
            f'- confidence intervals: {self.calc_ci}')

    def __repr__(self):
        return self.__str__()
    
    def fit(self, data):
        """Fit the data to the growth function.
        
        Parameters
        ----------
        data : numpy.ndarray
            Cumulative number of cumulative infectious cases.
        
        Returns
        -------
        None
        """
        if isinstance(data, (np.ndarray)):
            self.data = data.ravel()
        else:
            raise ValueError('Input data must be numpy.ndarray')
        if self.normalize:
            self.scaler = Scaler()
            y = self.scaler.fit_transform(data.reshape(-1, 1))
            x = normalize(np.arange(self.data.size))
        else: 
            y = self.data
            x = np.arange(self.data.size)

        if self.calc_ci and self.spread is not None:
            y_std = self.spread.ravel()
            abs_std = True
            if self.normalize:
                y_std = self.scaler.transform(y_std.reshape(-1, 1))
        else:
            y_std = np.ones_like(y)
            abs_std = False

        self.popt, self.pcov = curve_fit(self.function, x, y.ravel(),
            sigma=y_std.ravel(), absolute_sigma=abs_std, maxfev=3000)
        fitted = self.function(x, *self.popt)

        if self.calc_ci:
            self.perr = np.sqrt(np.diag(self.pcov))
            lower_bound = self.function(x, *(self.popt - self.perr))
            upper_bound = self.function(x, *(self.popt + self.perr))
            if self.normalize: 
                x = restore(x, np.arange(self.data.size))
                fitted = self.scaler.inverse_transform(fitted.reshape(-1, 1))
                lower_bound = self.scaler.inverse_transform(
                    lower_bound.reshape(-1, 1))
                upper_bound = self.scaler.inverse_transform(
                    upper_bound.reshape(-1, 1))
            fitted = np.r_[
                lower_bound.reshape(1, -1), 
                fitted.reshape(1, -1), 
                upper_bound.reshape(1, -1)]

        elif not self.calc_ci and self.normalize:
            x = restore(x, np.arange(self.data.size))
            fitted = self.scaler.inverse_transform(
                fitted.reshape(-1, 1)).ravel()
        self.x = x
        self.fitted = fitted
    
    @property
    def get_fitted(self):
        """Return fitted growth function over the independent variable
        where the data is measured.

        Parameters
        ----------
        None

        Returns
        -------
        x : numpy.ndarray
            The independent variable where the data is measured.
        fitted : numpy.ndarray
            Fitted growth function with lower and upper bound if
            `confidence_interval` is set to True.
        """
        if self.fitted is None:
            raise ValueError('Call `fit` method first.')
        return self.x, self.fitted

    @property
    def get_params(self):
        if self.popt is None:
            raise ValueError('No fitted parameters. Call `fit` method first.')
        return (self.popt)

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
                predicted = self.scaler.inverse_transform(
                    predicted.reshape(-1, 1))
                lower_bound = self.scaler.inverse_transform(
                    lower_bound.reshape(-1, 1))
                upper_bound = self.scaler.inverse_transform(
                    upper_bound.reshape(-1, 1))
            predicted = np.r_[
                lower_bound.reshape(1, -1), 
                predicted.reshape(1, -1), 
                upper_bound.reshape(1, -1)]
        elif not self.calc_ci and self.normalize:
            x_fut = restore(x_fut, np.arange(self.data.size))
            predicted = self.scaler.inverse_transform(
                predicted.reshape(-1, 1)).ravel()
        return x_fut, predicted