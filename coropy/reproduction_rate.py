################
# EXPERIMENTAL #
################

import datetime as dt
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .utils import moving_average


def _uncertainty_quantification(
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


def _estimate(confirmed_cases, averaging_period, symptoms_delay):
    """Return the time series estimate for reproduction number based on
    the number of the total number of positive confirmed cases.

    Parameters
    ----------
    averaging_period : int
        Moving average window.
    symptoms_delay : int
        Number of days from the first symptoms to confirmed infection.

    Returns
    -------
    numpy.ndarray
        Reproduction number values with the (n, ) shape where n is the
        total accumulated delay consisted of `averaging_period` and
        `symptoms_delay`.
    """
    if not isinstance(averaging_period, (int, )):
        averaging_period = int(averaging_period)
        warnings.warn(
            '`averaging_period` should be a positive integer.'
            'Automatic conversion float -> int is applied.')
    if not isinstance(symptoms_delay, (int, )):
        averaging_period = int(averaging_period)
        warnings.warn(
            '`symptoms_delay` should be a positive integer.'
            'Automatic conversion float -> int is applied.')
    confirmed_cases_bias = np.r_[0, confirmed_cases]
    new_daily_cases = np.diff(confirmed_cases_bias)
    smoothed_new_daily_cases = moving_average(new_daily_cases, averaging_period)
    R = np.divide(
        smoothed_new_daily_cases[symptoms_delay:], 
        smoothed_new_daily_cases[:-symptoms_delay],
        out=np.zeros(smoothed_new_daily_cases[:-symptoms_delay].size, dtype=float),
        where=smoothed_new_daily_cases[:-symptoms_delay]!=0)
    return R


def simulate(
    epidemics_start_date, 
    confirmed_cases,
    averaging_period=16,
    symptoms_delay=3,
    ci_plot=False,
    **kwargs):
    """Simulate and visualize the R0 with respect to the number of 
    confirmed infections daily."
    
    Parameters
    ----------
    epidemics_start_date : datetime.datetime
    confirmed_cases : numpy.ndarray
        The total number of confirmed infected COVID-19 cases per day.
    averaging_period : int, optional
        Parameter for smoothing the `confirmed_cases` array.
    symptoms_delay : int, optional
        Number of days between the infection and the confirmation
        (incubation period estimate).
    ci_plot : bool, optional
        Uncertainty quantification flag.
    **kwargs : dict, optional
        Keyword arguments related to uncertainty quantification.
    """
    delay = averaging_period + symptoms_delay

    R = _estimate(confirmed_cases, averaging_period, symptoms_delay)
    R_averaging_period = int(averaging_period / symptoms_delay)
    R_smoothed = moving_average(R, R_averaging_period)

    if ci_plot:
        try:
            sensitivity = kwargs['sensitivity']
            specificity = kwargs['specificity']
            daily_tests = kwargs['daily_tests']
            if sensitivity and specificity and daily_tests:
                lb_scaler, ub_scaler = _uncertainty_quantification(
                    sensitivity=sensitivity, 
                    specificity=specificity,
                    confirmed_cases=confirmed_cases,
                    daily_tests=daily_tests)
        except KeyError as e:
            warnings.warn(
            'Confidence interval could not be calculated. '
            'Invalid and/or missing keyword argument. '
            'Proceeding as `ci_plot=False`.')
            ci_plot = False
            
    epidemics_duration = confirmed_cases.size
    dates = mdates.drange(
        epidemics_start_date,
        epidemics_start_date + dt.timedelta(days=epidemics_duration), 
        dt.timedelta(days=1))
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(
        dates, confirmed_cases, 
        'b-', 
        label='Cumulative infectious cases')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylabel('Confirmed cases', color='b')
    ax1.legend()
    ax1.grid(None)

    ax2 = ax1.twinx()
    ax2.scatter(
        dates[:-delay+1], R, 
        color='r', edgecolor='black', 
        label='R values')
    ax2.plot(
        dates, np.ones(dates.shape), 
        'k--', 
        label='Critical R value')
    ax2.plot(
        dates[R_averaging_period:-delay+2], R_smoothed, 
        'r-', linewidth=2, 
        label='R averaged')
    if ci_plot:
        ax2.fill_between(
            dates[R_averaging_period:-delay+2],
            lb_scaler[R_averaging_period:-delay+2] * R_smoothed,
            ub_scaler[R_averaging_period:-delay+2] * R_smoothed,
            color='r', alpha=0.15, 
            label='95% CI')
    ax2.tick_params(labelcolor='r')
    ax2.set_ylabel('R values', color='r')
    ax2.legend()

    fig.gca().xaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m-%d'))
    fig.gca().xaxis.set_major_locator(
        mdates.DayLocator(interval=10))
    fig.autofmt_xdate()
    plt.show()