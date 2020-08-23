import datetime as dt

from matplotlib import dates as mdates
from matplotlib import rcParams 
import matplotlib.pyplot as plt 
import numpy as np


def latexconfig():
    """Configure aper ready plots - standard LaTeX configuration."""
    pgf_latex = {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "axes.labelsize": 10,
        "font.size": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.figsize": figsize(1.0),
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            ]
        }
    rcParams.update(pgf_latex)


def figsize(scale, nplots=1):
    """Golden ratio between the width and height: the ratio 
    is the same as the ratio of their sum to the width of 
    the figure. 
    
    width + height    height
    -------------- = --------
         width        width

    Props for the code goes to: 
    https://github.com/maziarraissi/PINNs/blob/master/Utilities/plotting.py

    Parameters
    ----------
    scale : float
        Figure scaled to `scale`.
    nplots : int
        Number of plots in a figure.

    Returns
    -------
    tuple
        Figure size.
    """
    fig_width_pt = 390.0 # LaTeX \textwidth
    inches_per_pt = 1.0/72.27
    golden_mean = (np.sqrt(5.0)-1.0)/2.0
    fig_width = fig_width_pt*inches_per_pt*scale
    fig_height = fig_width*golden_mean*nplots
    fig_size = [fig_width,fig_height]
    return fig_size


def plot_data(epidemics_start_date, confirmed_cases, recovered_cases, death_cases):
    """Plot time series data.
    
    Parameters
    ----------
    epidemics_start_date : datetime.datetime
        First day of the epidemics.
    confirmed_cases : numpy.ndarray
        Time series of the total number of infected individuals.
    recovered_cases : numpy.ndarray
        Time series of the total number of recovered individuals.
    death_cases : numpy.ndarray
        Time series of the total number of death cases.
    
    """
    removed_cases = recovered_cases + death_cases
    active_cases = confirmed_cases - removed_cases
    epidemics_end_date = epidemics_start_date + dt.timedelta(confirmed_cases.size)
    days = mdates.drange(epidemics_start_date, epidemics_end_date, dt.timedelta(days=1))
    fig = plt.figure(figsize=figsize(2, 3))
    axs = fig.subplots(nrows=3, ncols=1, sharex=True, squeeze=True)
    axs[0].plot(days, confirmed_cases, 'bo-', label='Total confirmed cases')
    axs[0].plot(days, recovered_cases, 'ro-', label='Total recovered cases')
    axs[0].legend()
    axs[0].grid()
    axs[0].set_ylabel('$N$')
    
    axs[1].plot(days, death_cases, 'bo-', label='Death cases')
    axs[1].legend()
    axs[1].grid()
    axs[1].set_ylabel('$N$')
    
    axs[2].plot(days, active_cases, 'bo-', label='Current active cases')
    axs[2].legend()
    axs[2].grid()
    axs[2].set_ylabel('$N$')
   
    _ = fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    _ = fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
    _ = plt.gcf().autofmt_xdate()
    
    plt.show()