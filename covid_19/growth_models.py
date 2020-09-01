import numpy as np 
from scipy.optimize import curve_fit 

from covid_19.utils import normalize, restore


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


_ci_dispatcher = {
    90: 1.645, 
    95: 1.960, 
    98: 2.326, 
    99: 2.576,
}


class GrowthCOVIDModel(object):
    """A class to fit an exponential model to given data."""
    def __init__(
        self, function, normalize=True, confidence_interval=False, **kwargs
        ):
        """Constructor.
        
        Parameters
        ----------
        function : str
            Growth curve.
        normalize : bool, optional
            Should the data be normalized to [0, 1] range.
        confidence_interval : bool, optional
            Generate CI using fitted params of a given function 
            +/- standard deviation.
        kwargs : dict, optional
            If `confidence_interval` flag is set to True and both
            kwargs are specified, the confidence intervals with be
            calculated using sensitivity as a valid accuracy measure.
            If confidence_interval flag is set to True but no kwargs
            (or just a single keyword argument) are given the
            `confidence_interval` will be calulated assuming there are
            no intrinsic errors on measured data.
            sensitivity : float
                Measure of the proportion of positives that are
                correctly identified (e.g., the percentage of sick
                people who are correctly identified as being infected).
            ci_level : int
                Confidence interval level. Supported values are 90, 95,
                98 and 99 using the standard normal distribution as the
                critical value.
        """
        try:
            self.function = _growth_model_dispatcher[function]
        except:
            raise ValueError('Try `exponential` or `logistic` growth models.')
        self.normalize = normalize
        self.ci = confidence_interval
        if len(kwargs) == 2: 
            for kw in kwargs.keys():
                if kw == 'sensitivity':
                    assert isinstance(kwargs[kw], (float, )), \
                        'Sensitivity has to be a floating point number.'
                    self.sensitivity = kwargs[kw]
                elif kw == 'ci_level':
                    assert kwargs[kw] in [90, 95, 98, 99], \
                        'Supported confidence interval levels are 90, 95, 98 \
                        and 99.'
                    self.ci_level = kwargs[kw]
                else:
                    raise ValueError('No such argument is supported.')
        else:
            self.sensitivity, self.ci_level = None, None
                    
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
            Fitted growth function with lower and upper bound if
            `confidence_interval` is set to True.
        """
        self.confirmed_cases = confirmed_cases

        if self.normalize:
            y = normalize(confirmed_cases)
        else: 
            y = confirmed_cases
    
        x = np.arange(confirmed_cases.size)
        self.popt, self.pcov = curve_fit(self.function, x, y)
        fitted = self.function(x, *self.popt) 
        if self.ci:
            if self.sensitivity and self.ci_level:
                std_sensitivity_err = np.sqrt(
                    np.divide(
                        (1 - self.sensitivity) * self.sensitivity, 
                        y, 
                        out=np.zeros_like(y), 
                        where=y!=0,
                        )
                )
                sensitivity_ci = _ci_dispatcher[self.ci_level] \
                                 * std_sensitivity_err
                lower_bound = (self.sensitivity - sensitivity_ci) * fitted
                upper_bound = (self.sensitivity + sensitivity_ci) * fitted
            else:
                self.perr = np.sqrt(np.diag(self.pcov))
                lower_bound = self.function(x, *(self.popt - self.perr))
                upper_bound = self.function(x, *(self.popt + self.perr))
        if self.normalize: 
            fitted = restore(fitted, confirmed_cases)
            if self.ci:
                lower_bound = restore(lower_bound, confirmed_cases)
                upper_bound = restore(upper_bound, confirmed_cases)
        if self.ci:
            fitted = np.r_[
                lower_bound.reshape(1, -1), 
                fitted.reshape(1, -1), 
                upper_bound.reshape(1, -1),
                ]
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
            Extrapolated data with lower and upper bound if 
            `confidence_interval` is set to True.
        """
        assert isinstance(n_days, (int,)), 'Number of days must be integer.'
        size = self.confirmed_cases.size
        x_future = np.arange(size-1, size+n_days)
        predicted = self.function(x_future, *self.popt)
        if self.ci:
            if self.sensitivity and self.ci_level:
                std_sensitivity_err = np.sqrt(
                    np.divide(
                        (1 - self.sensitivity) * self.sensitivity, 
                        predicted, 
                        out=np.zeros_like(predicted), 
                        where=predicted!=0,
                        )
                )
                sensitivity_ci = _ci_dispatcher[self.ci_level] \
                                 * std_sensitivity_err
                lower_bound = (self.sensitivity - sensitivity_ci) * predicted
                upper_bound = (self.sensitivity + sensitivity_ci) * predicted
            else:
                lower_bound = self.function(
                    x_future, *(self.popt - self.perr)
                    )
                upper_bound = self.function(
                    x_future, *(self.popt + self.perr)
                    )
        if self.normalize: 
            predicted = restore(predicted, self.confirmed_cases)
            if self.ci:
                lower_bound = restore(lower_bound, self.confirmed_cases)
                upper_bound = restore(upper_bound, self.confirmed_cases)
        if self.ci:
            predicted = np.r_[
                lower_bound.reshape(1, -1), 
                predicted.reshape(1, -1), 
                upper_bound.reshape(1, -1),
                ]
        return x_future, predicted