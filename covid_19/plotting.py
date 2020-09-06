import datetime as dt

from matplotlib import dates as mdates
from matplotlib import rcParams 
import matplotlib.pyplot as plt 
import numpy as np


def latexconfig():
    """Configure paper ready figures."""
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
    """Golden ratio between the width and height: the ratio is the same
    as the ratio of their sum to the width of the figure. 
    
    (width+height)/width = height/width

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


def plot_data(
    epidemics_start_date,
    confirmed_cases,
    recovered_cases,
    death_cases,
    daily_tests=None):
    """Plot the time series epidemiological statistics.
    
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
    daily_tests : numpy.ndarray, optional
        Time series of daily performed tests.
    """
    removed_cases = recovered_cases + death_cases
    active_cases = confirmed_cases - removed_cases
    epidemics_end_date = epidemics_start_date \
                         + dt.timedelta(confirmed_cases.size)
    days = mdates.drange(
        epidemics_start_date, epidemics_end_date, dt.timedelta(days=1)
        )
    if daily_tests is not None:
        n_figs = 4
    else:
        n_figs = 3
    fig = plt.figure(figsize=figsize(1.5, n_figs))
    axs = fig.subplots(nrows=n_figs, ncols=1, sharex=True, squeeze=True)
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
    if daily_tests is not None:
        axs[3].plot(days, daily_tests, 'bo-', label='Tests performed')
        axs[3].legend()
        axs[3].grid()
        axs[3].set_ylabel('$N$')
    _ = fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    _ = fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
    _ = plt.gcf().autofmt_xdate()
    plt.show()


def plot_loss(loss):
    """Plot the change in loss over time.
    
    Parameters
    ----------
    loss : list or numpy.ndarray
        Value of loss function with respect to iteration.
    """
    plt.plot(loss, 'b-')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()


def plot_compartmental_model_dynamics(
    epidemics_start_date,
    active_cases,
    I,
    recovered_cases,
    R,
    death_cases=None,
    D=None):
    """Fit the data to SEIR model and plot the SEIR simulation.
    
    Parameters
    ----------
    epidemics_start_date : datetime.datetime
        The first day of the epidemics outbreak.
    active_cases : numpy.ndarray
        Time series of currently active infected individuals.
    I : numpy.ndarray or list
        Fitted curve to the number of active cases.
    recovered_cases : numpy.ndarray
        Time series of removed (for SEIR) or recovered (for SEIRD)
        individuals.
    R : numpy.ndarray or list
        Fitted curve to the number of removed cases.
    death_cases : numpy.ndarray, optional
        Time series of deceased individuals.
    D : numpy.ndarray or list, optional
        Fitted curve to the number of death cases.
    """
    duration = active_cases.size
    end = epidemics_start_date + dt.timedelta(days=duration)    
    days = mdates.drange(
        epidemics_start_date, end, dt.timedelta(days=1))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    _ = fig.gca().xaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m-%d'))
    _ = fig.gca().xaxis.set_major_locator(
        mdates.DayLocator(interval=10))
    if I.ndim>1:
        ax.plot(days, I[0, :], 'r--')
        ax.plot(days, I[1, :], 'r-', label='$I(t)$')
        ax.plot(days, I[2, :], 'r--')
        ax.fill_between(days, I[0, :], I[2, :], color='r', alpha=0.1)
        ax.plot(days, R, 'g-', label='$R(t)$')
        if death_cases is not None and D is not None:
            ax.plot(days, D, 'b-', label='$D(t)$')
    else:
        ax.plot(days, I, 'r-', label='$I(t)$')
        ax.plot(days, R, 'g-', label='$R(t)$')
        if death_cases is not None and D is not None:
            ax.plot(days, D, 'b-', label='$D(t)$')
    ax.plot(
        days, active_cases, label='active cases',
        linestyle='None', marker='o', color='red', alpha=0.7)
    if death_cases is not None and D is not None:
        ax.plot(
            days, recovered_cases, label='recovered cases',
            linestyle='None', marker='o', color='green', alpha=0.7)
        ax.plot(
            days, death_cases, label='death cases',
            linestyle='None', marker='o', color='blue', alpha=0.7)
    else:
        ax.plot(
            days, recovered_cases, label='removed cases',
            linestyle='None', marker='o', color='green', alpha=0.7)
    plt.legend(loc='lower right')
    _ = plt.gcf().autofmt_xdate()
    plt.ylabel('$N$')
    plt.grid()
    plt.show()


def plot_compartmental_model_forecast(
    epidemics_start_date,
    active_cases,
    I,
    I_pred,
    recovered_cases,
    R,
    R_pred,
    death_cases=None,
    D=None,
    D_pred=None):
    """Plot the compartmental model forecast.
    
    Parameters
    ----------
    epidemics_start_date : datetime.datetime
        The first day of the epidemics outbreak.
    active_cases : numpy.ndarray
        Time series of currently active infected individuals.
    I : numpy.ndarray or list
        Fitted curve to the number of active cases.
    I_pred : numpy.ndarray or list
        Extrapolated I curve.
    recovered_cases : numpy.ndarray
        Time series of removed (for SEIR) or recovered (for SEIRD)
        individuals.
    R : numpy.ndarray or list
        Fitted curve to removed (for SEIR) or recovered (for SEIRD)
        individuals.
    R_pred : numpy.ndarray or list
        Extrapolated R curve.
    death_cases : numpy.ndarray, optional
        Time series of deceased individuals.
    D : numpy.ndarray or list, optional
        Fitted curve to the number of death cases.
    D_pred : numpy.ndarray or list, optional
        Extrapolated D curve.
    """
    sim_duration = active_cases.size
    forecast_duration = I_pred.size
    end_sim = epidemics_start_date + dt.timedelta(days=sim_duration)    
    sim_days = mdates.drange(
        epidemics_start_date, end_sim, dt.timedelta(days=1))
    start_forecast = end_sim + dt.timedelta(days=1)
    end_forecast = start_forecast \
                   + dt.timedelta(days=forecast_duration)
    forecast_days = mdates.drange(
        start_forecast, end_forecast, dt.timedelta(days=1))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    _ = fig.gca().xaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m-%d'))
    _ = fig.gca().xaxis.set_major_locator(
        mdates.DayLocator(interval=10))
    if I.ndim>1:
        ax.plot(sim_days, I[1, :], 'r-', label='$I(t)$')
    else:
        ax.plot(sim_days, I, 'r-', label='$I$ fitted')
    ax.plot(sim_days, R, 'g-', label='$R$ fitted')
    if death_cases is not None and D is not None and D_pred is not None:
        ax.plot(sim_days, D, 'b-', label='$D$ fitted')
    ax.plot(
        sim_days, active_cases, label='active cases',
        linestyle='None', marker='o', color='red', alpha=0.7)
    if death_cases is not None and D is not None and D_pred is not None:
        ax.plot(
        sim_days, recovered_cases, label='recovered cases',
        linestyle='None', marker='o', color='green', alpha=0.7)
        ax.plot(
            sim_days, death_cases, label='death cases',
            linestyle='None', marker='o', color='blue', alpha=0.7)
    else:
        ax.plot(
            sim_days, recovered_cases, label='removed cases',
            linestyle='None', marker='o', color='green', alpha=0.7)
    ax.plot(forecast_days, I_pred, 'r--', label='$I$ predicted')
    ax.plot(forecast_days, R_pred, 'g--', label='$R$ predicted')
    if death_cases is not None and D is not None and D_pred is not None:
        ax.plot(forecast_days, D_pred, 'b--', label='$D$ predicted')
    plt.legend(loc='lower right')
    _ = plt.gcf().autofmt_xdate()
    plt.ylabel('$N$')
    plt.grid()
    plt.show()


def plot_multiple_waves_simulation(
    epidemics_start_date,
    active_cases,
    I, 
    recovered_cases,
    R, 
    death_cases=None, 
    D=None):
    """Plot the recurrent epi oubreaks.
    
    Parameters
    ----------
    epidemics_start_date : datetime.datetime
        The first day of the epidemics outbreak.
    active_cases : numpy.ndarray
        Time series of currently active infected individuals.
    I : numpy.ndarray or list
        Fitted curve to the number of active cases.
    recovered_cases : numpy.ndarray
        Time series of recovered (for SEIRD) or removed (for SEIR)
        individuals.
    R : numpy.ndarray or list
        Fitted curve to the number of recovered (for SEIRD) or removed
        (for SEIR) individuals.
    death_cases : numpy.ndarray, optional
        Time series of deceased individuals.
    D : numpy.ndarray or list, optional
        Fitted curve to the number of death cases.
    """
    duration = active_cases.size
    end = epidemics_start_date + dt.timedelta(days=duration)    
    days = mdates.drange(epidemics_start_date, end, dt.timedelta(days=1))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    _ = fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    _ = fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
    ax.plot(days, I, 'r-', label='$I(t)$')
    ax.plot(
        days, active_cases, label='Active infections',
        linestyle='None', marker='o', color='red', alpha=0.7)
    ax.plot(days, R, 'g-', label='$R(t)$')
    ax.plot(
        days, recovered_cases, label='Recovered cases',
        linestyle='None', marker='o', color='green', alpha=0.7)
    if death_cases is not None and D is not None:
        ax.plot(days, D, 'b-', label='$D(t)$')
        ax.plot(
            days, death_cases, label='Deceased cases',
            linestyle='None', marker='o', color='blue', alpha=0.7)
    _ = plt.gcf().autofmt_xdate()
    plt.ylabel('$N$')
    plt.legend(loc='best')
    plt.grid()
    plt.show()