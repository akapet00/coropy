import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def configure_paper(golden_ratio=False, scale=1., n_plots=1):
    """Configure figures for paper.
    Parameters
    ----------
    golden_mean : bool, optional
        Golden ratio between the width and height is applied. `nplots`
        is set to 1 by default unless defined otherwise.
    scale : float, optional
        Figure scaled to `scale`.
    n_plots : float, optional
        Number of axes in the figure.

    Returns
    -------
    None
    """
    f_width_pt = 390.  # LaTeX \textwidth
    inches_per_pt = 1.0/72.27
    if golden_ratio:
        golden_scaler = (np.sqrt(5.0)-1.0)/2.0
    else:
        golden_scaler = 1
    f_width = f_width_pt * inches_per_pt * scale
    f_height = f_width * golden_scaler * n_plots
    
    plt.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'pgf.preamble': [
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}"],
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 12,
        'grid.linewidth': 0.7,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (f_width, f_height),
        'figure.max_open_warning': 120})


def plot_data(epidemics_start_date, confirmed_cases, recovered_cases,
    death_cases, daily_tests=None, paper_config=False):
    """Plot the time series epidemiological statistics.
    
    Parameters
    ----------
    epidemics_start_date : datetime.datetime
        First day of the epidemics.
    confirmed_cases : numpy.ndarray
        Cumulative number of confirmed positive infected cases.
    recovered_cases : numpy.ndarray
        Cumulative number of confirmed recoveries.
    death_cases : numpy.ndarray
        Cumulative number of confirmed deaths.
    daily_tests : numpy.ndarray, optional
        Time series of daily performed tests.
    paper_config : bool, optional
        Activate LaTeX mode.
    """
    removed_cases = recovered_cases + death_cases
    active_cases = confirmed_cases - removed_cases
    epidemics_end_date = epidemics_start_date \
                         + dt.timedelta(confirmed_cases.size)
    days = mdates.drange(
        epidemics_start_date, epidemics_end_date, dt.timedelta(days=1))
    if paper_config:
        configure_paper()
        fig = plt.figure(figsize=(9, 5))
    else:
        mpl.rcParams.update(mpl.rcParamsDefault)
        fig = plt.figure(figsize=(9, 5))
    axs = fig.subplots(nrows=2, ncols=2, sharex=True, squeeze=True)
    axs[0,0].plot(days, confirmed_cases, '.-', c='b',
        label='Total confirmed cases')
    axs[0,0].plot(days, recovered_cases, '.--', c='r',
        label='Total recovered cases')
    axs[0,0].legend()
    axs[0,0].grid()
    axs[0,0].set_ylabel('$N$')
    axs[1,0].plot(days, death_cases, '.-', c='b', label='Death cases')
    axs[1,0].legend()
    axs[1,0].grid()
    axs[1,0].set_ylabel('$N$')
    axs[0,1].plot(days, active_cases, '.-', c='b',
        label='Current active cases')
    axs[0,1].legend()
    axs[0,1].grid()
    axs[0,1].set_ylabel('$N$')
    if daily_tests is not None:
        axs[1,1].plot(days, daily_tests, '.-', c='b', label='Tests performed')
        axs[1,1].legend()
        axs[1,1].grid()
        axs[1,1].set_ylabel('$N$')
    _ = fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    _ = fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
    _ = plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


def avg_new_cases_over_total_cases(confirmed_cases, avg_period=7,
    paper_config=False):
    """Plot averaged new cases over total cases in time.

    Parameters
    ----------
    confirmed_cases : numpy.ndarray
        Cumulative number of confirmed positive infected cases.
    avg_period : int, optional
        Averaging period, number of days.
    paper_config : bool, optional
        Activate LaTeX mode.
    """
    iterations = confirmed_cases.size
    cases_periodically = []
    epoch = 0
    while epoch < iterations:
        cases_periodically.append(confirmed_cases[epoch])
        epoch += avg_period
    cases_periodically = np.array(cases_periodically)
    
    new_cases = []
    for epoch, csp in enumerate(cases_periodically):
        if epoch==0:
            new_cases.append(csp)
        else: 
            new_cases.append(cases_periodically[epoch] \
                             - cases_periodically[epoch-1])
    new_cases = np.array(new_cases)

    _x = np.linspace(0, np.max(cases_periodically))
    if paper_config:
        configure_paper(golden_ratio=True)
        fig = plt.figure()
    else:
        mpl.rcParams.update(mpl.rcParamsDefault)
        fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog(cases_periodically, new_cases, '-', c='b', label='data')
    ax.plot(_x, _x, ':', c='b', label='theoretical exp') # exp growth
    ax.set_xlabel(f'{avg_period}-day average total confirmed cases')
    ax.set_ylabel('Daily confirmed cases')
    ax.grid()
    ax.legend()
    plt.show()