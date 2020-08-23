import datetime as dt

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates

from covid_19.utils import normalize, restore, moving_average, train_test_split
from covid_19.plotting import plot_data, figsize, latexconfig
from covid_19.growth_models import GrowthCOVIDModel
from covid_19.compartmental_models import SEIRModel


def initial_growth_fit(function, eff_date, confirmed_cases, n_days):
    """
    Fit the data to exponential function and plot n_days forecast.
        
    Parameters
    ----------
    function : str
        Growth curve.
    eff_date : datetime.datetime
        Start date of simulation.
    confirmed_cases : numpy.ndarray
        Number of confirmed infected COVID-19 cases per day since eff_date.
    n_days : int
        Number of days to extrapolate.
    """
    date_list = [eff_date + dt.timedelta(days=i) for i in range(confirmed_cases.size)]
    offset = np.min(confirmed_cases)
    confirmed_cases_adjusted = confirmed_cases - offset
    exp_model = GrowthCOVIDModel(function='exponential', normalize=True)
    _, fitted_curve = exp_model.fit(confirmed_cases_adjusted)
    _, predicted_curve = exp_model.predict(n_days)
    date_list_future = [date_list[-1] + dt.timedelta(days=i) for i in range(predicted_curve.size)]
    
    plt.figure(figsize=figsize(1.5, 1))
    plt.plot(date_list, confirmed_cases_adjusted + offset, color='blue', linestyle='-', marker='o', label='Confirmed cases')
    plt.plot(date_list, fitted_curve + offset, color='red', linestyle='-', marker='None', label='Exponential fit')
    plt.plot(date_list_future, predicted_curve + offset, color='red', linestyle='--', marker='o',  label='Extrapolated curve')
    props = dict(boxstyle='round', facecolor='lavender', alpha=0.7)
    for vals in zip(date_list_future[1:], predicted_curve[1:] + offset):
        plt.text(vals[0], vals[1] - 175, str(int(vals[1])), verticalalignment='top', bbox=props)
    plt.legend()
    plt.grid()
    plt.gcf().autofmt_xdate()
    plt.show()


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
            new_cases.append(confirmed_cases_periodically[epoch] - confirmed_cases_periodically[epoch-1])
    new_cases = np.array(new_cases)

    _x = np.linspace(0, np.max(confirmed_cases_periodically))

    fig = plt.figure(figsize=figsize(1,1))
    ax = fig.add_subplot(111)
    ax.loglog(confirmed_cases_periodically, new_cases, 'b-', label='Measured data')
    ax.plot(_x, _x, 'k:', label='Exponential growth') # exponential growth ground truth
    ax.set_xlabel('$N_{total}$')
    ax.set_ylabel('$N_{new}$')
    ax.grid()
    ax.legend()
    plt.show()


def seir_simulation(active_cases, removed_cases, initial_conditions, split_ratio, epidemics_start_date):
    """Plot SEIR forecast and return R0 and loss values.
    
    Parameters
    ----------
    active_cases: numpy.ndarray
        Time series of currently active infected individuals.  
    removed_cases: numpy.ndarray
        Time series of recovered+deceased individuals.
    initial_conditions: list
        Values of S, E, I and R at the first day.
    split_ratio : float
        Ratio for data split.
    epidemics_start_date : datetime.datetime
        First day of the observed epidemic wave.
    
    Returns
    -------
    float
        Reproduction number.
        
    list
        Loss values during the optimization procedure.
    """
    train_active_cases, test_active_cases = train_test_split(active_cases, split_ratio)
    train_removed_cases, test_removed_cases = train_test_split(removed_cases, split_ratio)
    
    n_days = train_active_cases.size * 3

    seir_model = SEIRModel()
    
    (beta, delta, alpha, gamma), loss = seir_model.fit(
        train_active_cases, train_removed_cases, initial_conditions
    )
    R_eff = beta / (alpha+gamma)
    (S, E, I, R) = seir_model.predict(n_days)

    end = epidemics_start_date + dt.timedelta(days=n_days)    
    days = mdates.drange(epidemics_start_date, end, dt.timedelta(days=1))

    fig = plt.figure(figsize=figsize(1, 1))
    ax = fig.add_subplot(111)
    _ = fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    _ = fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
    ax.plot(days, S, 'k-', label='$S(t)$')
    ax.plot(days, E, 'b-', label='$E(t)$')
    ax.plot(days, I, 'r-', label='$I(t)$')
    ax.plot(days, R, 'g-', label='$R(t)$')
    ax.plot(
        days[:train_active_cases.size],
        train_active_cases, 
        linestyle='None', 
        marker='o', 
        color='red', 
        alpha=0.7, 
        label='active cases',
    )
    ax.plot(
        days[:train_removed_cases.size],
        train_removed_cases, 
        linestyle='None', 
        marker='o', 
        color='green', 
        alpha=0.7, 
        label='removed cases',
    )
    if test_active_cases.size > 0:
        ax.scatter(
            days[len(train_active_cases):len(train_active_cases)+len(test_active_cases)],
            test_active_cases,
            facecolors='none', 
            edgecolors='r', 
            linestyle='None',
            label='test infected',
        )
        ax.scatter(
            days[len(train_removed_cases):len(train_removed_cases)+len(test_removed_cases)],
            test_removed_cases,
            facecolors='none', 
            edgecolors='g', 
            linestyle='None',
            label='test removed',
        )
        plt.axis([mdates.date2num(epidemics_start_date - dt.timedelta(days=1)), days[active_cases.size + 1], -100, 1750])
        _ = fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.legend(loc='best')
    else:
        plt.legend(loc='lower right')
    _ = plt.gcf().autofmt_xdate()
    plt.ylabel('$N$')
    plt.grid()
    plt.show()
    return R_eff, loss


def main():
    # data
    confirmed_cases = np.loadtxt('data/confirmed_cases.dat')
    recovered_cases = np.loadtxt('data/recovered_cases.dat')
    death_cases = np.loadtxt('data/death_cases.dat')
    removed_cases = recovered_cases + death_cases
    active_cases = confirmed_cases - removed_cases

    # 1st wave data
    start_date_1 = dt.datetime(2020, 2, 26)
    end_date_1 = dt.datetime(2020, 6, 5)
    diff = abs((end_date_1 - start_date_1).days)
    confirmed_cases_1 = confirmed_cases[:diff+1]
    active_cases_1 = active_cases[:diff+1]
    removed_cases_1 = recovered_cases[:diff+1]

    # 2nd wave data
    start_date_2 = dt.datetime(2020, 6, 6)
    confirmed_cases_2 = confirmed_cases[diff+1:]
    active_cases_2 = active_cases[diff+1:]
    removed_cases_2 = removed_cases[diff+1:] - removed_cases[diff+1] # scaled

    # full data visualization
    plot_data(
        start_date_1, 
        confirmed_cases, 
        recovered_cases, 
        death_cases,
    )

    # full data wave averaged_new_cases_v_total_cases
    averaged_new_cases_v_total_cases(confirmed_cases, 7)

    # 1st wave SEIR simulation
    S0 = 2300
    E0 = 1
    I0 = active_cases_1[0]
    R0 = removed_cases_1[0]
    R_eff, loss = seir_simulation(
        active_cases=active_cases_1, 
        removed_cases=removed_cases_1, 
        initial_conditions=(S0, E0, I0, R0),
        split_ratio=1.,
        epidemics_start_date=start_date_1,
    )
    plt.plot(np.sqrt(loss), 'b-')
    plt.xlabel('Iterations')
    plt.ylabel(r'$\sqrt{loss}$')
    plt.grid()
    plt.show()

    # 2nd wave exponential fit
    eff_date = dt.datetime(2020, 8, 1)
    diff = abs((eff_date - start_date_1).days)
    n_days = 7
    initial_growth_fit('exponential', eff_date, confirmed_cases[diff+1:], n_days)


if __name__ == "__main__":
    # latexconfig()
    main()