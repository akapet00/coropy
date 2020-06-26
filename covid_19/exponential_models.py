import numpy as np 
from scipy.optimize import curve_fit 
from covid_19.utils import normalize, restore

def _exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def _logistic_func(x, a, b, c, d):
    return a / (1 + np.exp(-c * (x - d))) + b

class Model(object):
    """General class."""
    
    def __init__(self, normalize=True):
        self.normalize = normalize

class ExponentialModel(Model):
    """Fit data to exponential function.""" 
        
    def fit(self, x, confirmed_cases):
        self.x = x 
        self.confirmed_cases = confirmed_cases

        if self.normalize:
            y = normalize(confirmed_cases)
        else: y = confirmed_cases

        self.popt, self.pcov = curve_fit(_exp_func, x, y)
        fitted = _exp_func(x, *self.popt) 
        if self.normalize: fitted = restore(fitted, confirmed_cases)
        return fitted, (self.popt, self.pcov) 

    def predict(self, n_days):
        x_future = np.linspace(len(self.confirmed_cases)-1, 
                                len(self.confirmed_cases) + n_days-1)
        predict = _exp_func(x_future, *self.popt)
        if self.normalize: predict = restore(predict, self.confirmed_cases)
        return x_future, predict

class LogisticModel(Model):
    """Fit data to logistic function."""

    def fit(self, x, confirmed_cases):
        self.x = x 
        self.confirmed_cases = confirmed_cases

        if self.normalize:
            y = normalize(confirmed_cases)
        else: y = confirmed_cases

        self.popt, self.pcov = curve_fit(_logistic_func, x, y)
        fitted = _logistic_func(x, *self.popt) 
        if self.normalize: fitted = restore(fitted, confirmed_cases)
        return fitted, (self.popt, self.pcov) 

    def predict(self, n_days):
        x_future = np.linspace(len(self.confirmed_cases)-1, 
                                len(self.confirmed_cases) + n_days-1)
        predict = _logistic_func(x_future, *self.popt)
        if self.normalize: predict = restore(predict, self.confirmed_cases)
        return x_future, predict

        

