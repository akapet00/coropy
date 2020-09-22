import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .plotting import figsize
from .plotting import plot_data
from .plotting import plot_loss
from .plotting import plot_compartmental_model_dynamics
from .plotting import plot_compartmental_model_forecast
from .plotting import plot_multiple_waves_simulation
from .growth_models import GrowthCOVIDModel
from .compartmental_models import SEIRModel, SEIRDModel


def averaged_new_cases_v_total_cases(confirmed_cases, avg_period=7):
    """Plot averaged new cases over total cases in time.

    Parameters
    ----------
    confirmed_cases : numpy.ndarray
        Number of confirmed infected COVID-19 cases per day.
    avg_period : int, optional
        Averaging period, number of days.
    """
    iterations = confirmed_cases.size
    confirmed_cases_periodically = []
    epoch = 0
    while epoch < iterations:
        confirmed_cases_periodically.append(confirmed_cases[epoch])
        epoch += avg_period
    confirmed_cases_periodically = np.array(confirmed_cases_periodically)
    
    new_cases = []
    for epoch, csp in enumerate(confirmed_cases_periodically):
        if epoch==0:
            new_cases.append(csp)
        else: 
            new_cases.append(
                confirmed_cases_periodically[epoch] \
                - confirmed_cases_periodically[epoch-1])
    new_cases = np.array(new_cases)

    _x = np.linspace(0, np.max(confirmed_cases_periodically))
    fig = plt.figure(figsize=figsize(1.5, 1))
    ax = fig.add_subplot(111)
    ax.loglog(
        confirmed_cases_periodically, new_cases, 'b-', label='Measured data')
    ax.plot(_x, _x, 'k:', label='Exponential growth') # exponential growth
    ax.set_xlabel(f'{avg_period}-day average total confirmed cases')
    ax.set_ylabel('New confirmed cases')
    ax.grid()
    ax.legend()
    plt.show()


def initial_growth(
    function, 
    eff_date, 
    confirmed_cases, 
    normalize_data, 
    n_days, 
    plot_confidence_intervals=False, 
    **kwargs):
    """
    Fit the data to the growth function and plot n_days forecast.
        
    Parameters
    ----------
    function : str
        Growth curve.
    eff_date : datetime.datetime
        Start date of simulation.
    confirmed_cases : numpy.ndarray
        Number of confirmed infected COVID-19 cases per day since 
        `eff_date`.
    n_days : int
        Number of days to extrapolate.
    plot_confidence_intervals : bool, optional
        Plot confidence intervals.
    kwargs
        If specified, keyword arguments are passed to 
        `GrowthCOVIDModel` when `confidence_interval`=True.
    """
    confirmed_cases_adjusted = confirmed_cases
    exp_model = GrowthCOVIDModel(
        function=function,
        normalize=normalize_data,
        confidence_interval=plot_confidence_intervals,
        **kwargs)
    x, fitted_curve = exp_model.fit(confirmed_cases)
    date_list = [
        eff_date + dt.timedelta(days=i) for i in range(x.size)]
    x_future, predicted_curve = exp_model.predict(n_days)
    date_list_future = [
        date_list[-1] + dt.timedelta(days=i) 
        for i in range(x_future.size + 1)]
    date_list_future = date_list_future[1:]
    props = dict(boxstyle='round', facecolor='lavender', alpha=1.0)
      
    plt.figure(figsize=(12, 6))
    plt.plot(
        date_list, confirmed_cases_adjusted, label='confirmed cases',
        color='blue', linestyle='-', marker='o')
    if plot_confidence_intervals:
        plt.plot(
            date_list, fitted_curve[0, :], label=f'lower/upper bound',
            color='red', linestyle='--', marker='None')
        plt.plot(
            date_list, fitted_curve[1, :], label=f'{function} fit',
            color='red', linestyle='-', marker='None')
        plt.plot(
            date_list, fitted_curve[2, :],
            color='red', linestyle='--', marker='None')
        plt.fill_between(
            date_list, fitted_curve[0, :], fitted_curve[2, :], 
            color='red', alpha=0.1)
        
        plt.plot(
            date_list_future, predicted_curve[0, :], 
            color='red', linestyle='--', marker='None')
        plt.plot(
            date_list_future, predicted_curve[1, :], label='extrapolated',
            color='red', linestyle='-', marker='o')
        plt.plot(
            date_list_future, predicted_curve[2, :],
            color='red', linestyle='--', marker='None')
        plt.fill_between(
            date_list_future, predicted_curve[0, :], predicted_curve[2, :],
            color='red', alpha=0.1)
        
        for vals in zip(date_list_future, predicted_curve[1, :]):
            plt.text(
                vals[0], vals[1] - 500, str(int(vals[1])), 
                verticalalignment='top', bbox=props)
    else:
        plt.plot(
            date_list, fitted_curve, label=f'{function} fit',
            color='red', linestyle='-', marker='None')
        plt.plot(
            date_list_future, predicted_curve, label='extrapolated',
            color='red', linestyle='--', marker='o')
        for vals in zip(date_list_future, predicted_curve):
            plt.text(
                vals[0], vals[1] - 500, str(int(vals[1])), 
                verticalalignment='top', bbox=props)
    plt.legend()
    plt.grid()
    plt.gcf().autofmt_xdate()
    plt.show()

    
def seir_dynamics(
    active_cases,
    removed_cases,
    initial_conditions,
    epidemics_start_date,
    plot_sim=False,
    plot_l=False,
    sensitivity=None,
    specificity=None,
    new_positives=None,
    total_tests=None):
    """Simulate SEIR model.
    
    Parameters
    ----------
    active_cases : numpy.ndarray
        Time series of currently active infected individuals.  
    removed_cases : numpy.ndarray
        Time series of recovered+deceased individuals.
    initial_conditions: list
        Values of S, E, I and R at the first day.
    epidemics_start_date : datetime.datetime
        First day of the observed epidemic wave.
    plot_sim : bool, optional
        Indicates if the simulation is going to be plotted.
    plot_l: bool, optional
        Indicates if loss values are going to be plotted.
    sensitivity : float, optional
        Test sensitivity.
    specificity : float, optional
        Test specificity.
    new_positives : numpy.ndarray, optional
        Time series of new daily infected individuals.
    total_tests : numpy.ndarray, optional
        Time series of new daily completed tests.
    
    Returns
    -------
    (S, E, I, R) : tuple
        Values of S, E, I and R fitted curves.
    seir_model : covid_19.CompartmentalModel
        Fitted compartmental epi model.
    loss : list
        Loss values during the optimization procedure.
    """    
    seir_model = SEIRModel(
        loss_fn='mse',
        sensitivity=sensitivity,
        specificity=specificity,
        new_positives=new_positives,
        total_tests=total_tests)
    
    (beta, delta, alpha, gamma), loss = seir_model.fit(
        active_cases, removed_cases, initial_conditions)
    (S, E, I, R) = seir_model.simulate()
    
    if plot_sim:
        plot_compartmental_model_dynamics(
            epidemics_start_date, active_cases, I, removed_cases, R)
    if plot_l:
        plot_loss(loss)
    return (S, E, I, R), seir_model, loss


def seird_dynamics(
    active_cases,
    recovered_cases,
    death_cases,
    initial_conditions,
    epidemics_start_date,
    plot_sim=False,
    plot_l=False,
    sensitivity=None,
    specificity=None,
    new_positives=None,
    total_tests=None):
    """Simulate SEIRD model.
    
    Parameters
    ----------
    active_cases : numpy.ndarray
        Time series of currently active infected individuals.  
    recovered_cases : numpy.ndarray
        Time series of recovered individuals.
    death_cases : numpy.ndarray
        Time series of deceased individuals.
    initial_conditions: list
        Values of S, E, I and R at the first day.
    epidemics_start_date : datetime.datetime
        First day of the observed epidemic wave.
    plot_sim : bool, optional
        Indicates if the simulation is going to be plotted.
    plot_l: bool, optional
        Indicates if loss values are going to be plotted.
    sensitivity : float, optional
        Test sensitivity.
    specificity : float, optional
        Test specificity.
    new_positives : numpy.ndarray, optional
        Time series of new daily infected individuals.
    total_tests : numpy.ndarray, optional
        Time series of new daily completed tests.
    
    Returns
    -------
    (S, E, I, R, D) : tuple
        Values of S, E, I, R and D fitted curves.
    seird_model : covid_19.CompartmentalModel
        Fitted compartmental epi model.
    loss : list
        Loss values during the optimization procedure.        
    """    
    seird_model = SEIRDModel(
        loss_fn='mse',
        sensitivity=sensitivity,
        specificity=specificity,
        new_positives=new_positives,
        total_tests=total_tests)
    
    (beta, alpha, gamma, mu), loss = seird_model.fit(
        active_cases,
        recovered_cases,
        death_cases, 
        initial_conditions)
    (S, E, I, R, D) = seird_model.simulate()
    
    if plot_sim:
        plot_compartmental_model_dynamics(
            epidemics_start_date,
            active_cases,
            I,
            recovered_cases,
            R,
            death_cases,
            D)
    if plot_l:
        plot_loss(loss)
    
    return (S, E, I, R, D), seird_model, loss


def seir_multiple_waves(
    first_wave_eff_population,
    eff_dates,
    active_cases, 
    removed_cases,
    plot_sim):
    """Simulate multiple epidemiological waves phenomena.
    
    Parameters
    ----------
    first_wave_eff_population : int
        Number of effective susceptible population.
    eff_dates : list
        List of datetime.datetime formated effective dates which mark
        the beginning of an increase in the number of infected
        individuals (first day of the new wave).
    active_cases : numpy.ndarray
        Time series of currently active infected individuals.  
    removed_cases : numpy.ndarray
        Time series of removed individuals.
    plot_sim : bool, optional
        Indicates if the simulation will be plotted.
    
    Returns
    -------
    tuple
        Tuple which contains values of the fitted S, E, I and R curves.
    """
    eff_population_scaler = 1
    S0 = first_wave_eff_population * eff_population_scaler
    initial_conditions = [
        S0,
        3 * active_cases[0],
        active_cases[0],
        removed_cases[0]]
    S_tot, E_tot, I_tot, R_tot = [], [], [], []
    start_idx = 0
    # past wave(s) simulation
    for start_date, end_date in zip(eff_dates[:-1], eff_dates[1:]):
        end_idx = start_idx+abs((end_date - start_date).days)
    
        seir_model = SEIRModel()
        _, _ = seir_model.fit(
            active_cases[start_idx:end_idx], 
            removed_cases[start_idx:end_idx],
            initial_conditions=initial_conditions)
        (S, E, I, R) = seir_model.simulate()
        S_tot.extend(S.tolist())
        E_tot.extend(E.tolist())
        I_tot.extend(I.tolist())
        R_tot.extend(R.tolist())
        
        # update initial conditions
        eff_population_scaler += 1
        S0 = S0 * eff_population_scaler
        initial_conditions = [
            S0,
            3 * I[-1],
            I[-1],
            R[-1]]
        # update indexing
        start_idx = end_idx
        
    # current wave simulation   
    seir_model = SEIRModel()
    _, _ = seir_model.fit(
        active_cases[start_idx:], 
        removed_cases[start_idx:],
        initial_conditions=initial_conditions)
    (S, E, I, R) = seir_model.simulate()
    S_tot.extend(S.tolist())
    E_tot.extend(E.tolist())
    I_tot.extend(I.tolist())
    R_tot.extend(R.tolist())
    
    if plot_sim:
        plot_multiple_waves_simulation(
            epidemics_start_date=eff_dates[0],
            active_cases=active_cases,
            I=I_tot, 
            recovered_cases=removed_cases,
            R=R_tot,
            death_cases=None, 
            D=None)
    return (S_tot, E_tot, I_tot, R_tot)


def seird_multiple_waves(
    first_wave_eff_population,
    eff_dates,
    active_cases, 
    recovered_cases, 
    death_cases,
    plot_sim=False):
    """Simulate multiple epidemiological waves phenomena.
    
    Parameters
    ----------
    first_wave_eff_population : int
        Number of effective susceptible population.
    eff_dates : list
        List of datetime.datetime formated effective dates 
        which mark the beginning of an increase in the number
        of infected individuals (first day of the new wave).
    active_cases : numpy.ndarray
        Time series of currently active infected individuals.  
    recovered_cases : numpy.ndarray
        Time series of recovered individuals.
    death_cases : numpy.ndarray
        Time series of deceased individuals.
    plot_sim : bool, optional
        Indicates if the simulation will be plotted.
    
    Returns
    -------
    tuple
        Tuple which contains values of the fitted S, E, I, R 
        and D curves.
    """
    eff_population_scaler = 1
    S0 = first_wave_eff_population * eff_population_scaler
    initial_conditions = [
        S0,
        3 * active_cases[0],
        active_cases[0],
        recovered_cases[0],
        death_cases[0]]
    S_tot, E_tot, I_tot, R_tot, D_tot = [], [], [], [], []
    start_idx = 0
    # past wave(s) simulation
    for start_date, end_date in zip(eff_dates[:-1], eff_dates[1:]):
        end_idx = start_idx+abs((end_date - start_date).days)
    
        seird_model = SEIRDModel()
        _, _ = seird_model.fit(
            active_cases[start_idx:end_idx], 
            recovered_cases[start_idx:end_idx], 
            death_cases[start_idx:end_idx], 
            initial_conditions=initial_conditions)
        (S, E, I, R, D) = seird_model.simulate()
        S_tot.extend(S.tolist())
        E_tot.extend(E.tolist())
        I_tot.extend(I.tolist())
        R_tot.extend(R.tolist())
        D_tot.extend(D.tolist())
        
        # update initial conditions
        eff_population_scaler += 1
        S0 = S0 * eff_population_scaler
        initial_conditions = [
            S0,
            3 * I[-1],
            I[-1],
            R[-1],
            D[-1]]
        # update indexing
        start_idx = end_idx
        
    # current wave simulation   
    seird_model = SEIRDModel()
    _, _ = seird_model.fit(
        active_cases[start_idx:], 
        recovered_cases[start_idx:], 
        death_cases[start_idx:], 
        initial_conditions=initial_conditions)
    (S, E, I, R, D) = seird_model.simulate()
    S_tot.extend(S.tolist())
    E_tot.extend(E.tolist())
    I_tot.extend(I.tolist())
    R_tot.extend(R.tolist())
    D_tot.extend(D.tolist())
    
    if plot_sim:
        plot_multiple_waves_simulation(
            epidemics_start_date=eff_dates[0],
            active_cases=active_cases,
            I=I_tot, 
            recovered_cases=recovered_cases,
            R=R_tot, 
            death_cases=death_cases, 
            D=D_tot)
    return (S_tot, E_tot, I_tot, R_tot, D_tot)