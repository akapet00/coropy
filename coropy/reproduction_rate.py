################
# EXPERIMENTAL #
################

import datetime as dt
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .utils import moving_average


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


def simulate(epidemics_start_date, confirmed_cases, averaging_period=16,
    symptoms_delay=3):
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
    """
    delay = averaging_period + symptoms_delay

    R = _estimate(confirmed_cases, averaging_period, symptoms_delay)
    R_averaging_period = int(averaging_period / symptoms_delay)
    R_smoothed = moving_average(R, R_averaging_period)
            
    epidemics_duration = confirmed_cases.size
    dates = mdates.drange(epidemics_start_date,
        epidemics_start_date + dt.timedelta(days=epidemics_duration),
        dt.timedelta(days=1))
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(dates, confirmed_cases, 'b-', label='Cumalitive cases')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylabel('Confirmed cases', color='b')
    ax1.legend()
    ax1.grid(None)

    ax2 = ax1.twinx()
    ax2.scatter(dates[:-delay+1], R, color='r', edgecolor='black',
        label='R values')
    ax2.plot(dates, np.ones(dates.shape), 'k--', label='Critical value')
    ax2.plot(dates[R_averaging_period:-delay+2], R_smoothed, 'r-',
        linewidth=2, label='R averaged')
    ax2.tick_params(labelcolor='r')
    ax2.set_ylabel('R values', color='r')
    ax2.legend()

    fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
    fig.autofmt_xdate()
    plt.show()