################
# EXPERIMENTAL #
################

import warnings

import numpy as np

from covid_19.utils import moving_average


def uncertainty_quantification(
    sensitivity, specificity, confirmed_cases, daily_tests):
    """Return the lower and the upper bound scaler for the reproduction
    number in time in the 95% confidence interval.
    
    Parameters
    ----------
    sensitivity : float or numpy.ndarray
        Test sensitivity. If the sensitivity is different in
        different intervals, it should be stored in the array-like
        format where the length is the same as the length of the
        data.
    specificity : float or numpy.ndarray
        Test specificity. If the sensitivity is different in
        different intervals, it should be stored in the array-like
        format where the length is the same as the length of the
        data.
    confirmed_cases : numpy.ndarray
        The total number of confirmed infected COVID-19 cases per day.     
    daily_tests : numpy.ndarray
        Daily number of tests.
    
    Returns
    -------
    lower_bound_scaler : numpy.ndarray
        Array of shape (n, ) where n is the duration of modeled
        epidemics in days with values of lower CI bound scaler. 
    upper_bound_scaler : numpy.ndarray
        Array of shape (n, ) where n is the duration of modeled
        epidemics in days with values of upper CI bound scaler. 
    """
    positives = np.diff(confirmed_cases)
    std_sensitivity_err = np.sqrt(np.divide(
        (1 - sensitivity) * sensitivity, 
        positives, 
        out=np.zeros(positives.shape, dtype=float), 
        where=positives!=0,
    ))
    sensitivity_ci = 1.96 * std_sensitivity_err
    lower_bound_sensitivity = np.abs(sensitivity - sensitivity_ci)
    lower_bound_true_positives = lower_bound_sensitivity * positives
    lower_bound_cumulative_cases = np.cumsum(lower_bound_true_positives) \
                                    + confirmed_cases[0]
    
    daily_tests = np.concatenate((
        np.array([daily_tests[0]]), daily_tests[:-1]))
    negatives = daily_tests[1:] - positives
    std_specificity_err = np.sqrt(np.divide(
        (1 - specificity) * specificity,
        negatives,
        out=np.zeros(negatives.shape, dtype=float), 
        where=negatives!=0,
    ))
    specificity_ci = 1.96 * std_specificity_err
    upper_bound_specificity = np.abs(specificity + specificity_ci)
    upper_bound_true_negatives = upper_bound_specificity * negatives
    upper_bound_false_negatives = negatives - upper_bound_true_negatives
    upper_bound_true_positives = upper_bound_false_negatives + positives
    upper_bound_cumulative_cases = np.cumsum(upper_bound_true_positives) \
                                    + confirmed_cases[0]

    lower_bound_cumulative_cases = np.concatenate((
        np.array([confirmed_cases[0]]), lower_bound_cumulative_cases))
    upper_bound_cumulative_cases = np.concatenate((
        np.array([confirmed_cases[0]]), upper_bound_cumulative_cases))

    lower_bound_scaler = lower_bound_cumulative_cases / confirmed_cases
    lower_bound_scaler[lower_bound_scaler>1]=1
    upper_bound_scaler = upper_bound_cumulative_cases / confirmed_cases
    upper_bound_scaler[upper_bound_scaler<1]=1
    return lower_bound_scaler, upper_bound_scaler


def estimate(confirmed_cases, averaging_period=7, testing_delay=5):
    """Return the time series estimate for reproduction number based on
    the number of the total number of positive confirmed cases.

    Parameters
    ----------
    averaging_period : int
        Moving average window.
    testing_delay : int
        Number of days from the first symptom to confirmed infection.

    Returns
    -------
    numpy.ndarray
        Reproduction number values with the (n, ) shape where n is the
        total accumulated delay consisted of `averagin_period` and
        `testing_delay`.
    """
    if not isinstance(averaging_period, (int, )):
        averaging_period = int(averaging_period)
        warnings.warn(
            '`averaging_period` should be a positive integer.'
            'Automatic conversion float -> int is applied.')
    if not isinstance(testing_delay, (int, )):
        averaging_period = int(averaging_period)
        warnings.warn(
            '`testing_delay` should be a positive integer.'
            'Automatic conversion float -> int is applied.')
    confirmed_cases_bias = np.r_[0, confirmed_cases]
    new_daily_cases = np.diff(confirmed_cases_bias)
    smoothed_new_daily_cases = moving_average(new_daily_cases, averaging_period)
    R = np.divide(
        smoothed_new_daily_cases[testing_delay:], 
        smoothed_new_daily_cases[:-testing_delay],
        out=np.zeros(smoothed_new_daily_cases[:-testing_delay].size, dtype=float),
        where=smoothed_new_daily_cases[:-testing_delay]!=0)
    return R