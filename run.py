import datetime as dt
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score

from covid_19.utils import normalize, restore, moving_average, train_test_split
from covid_19.plotting import plotData, figsize, latexconfig
from covid_19.exponential_models import ExponentialModel, LogisticModel
from covid_19.compartmental_models import SIR

def exp_fit(x, train_confirmed_cases, test_confirmed_cases, n_future_days):
    fig, ax = plotData(x, train_confirmed_cases, False, 1.5)

    exp_model = ExponentialModel()
    fitted, _ = exp_model.fit(x, train_confirmed_cases)
    ax.plot(x, fitted, 'r-', label='exponential fit')

    # extrapolation 
    _x, preds = exp_model.predict(n_future_days)
    ax.plot(_x, preds, 'r--', label='predictions')
    ax.plot(np.arange(len(train_confirmed_cases), 
                      len(train_confirmed_cases)+len(test_confirmed_cases), 1),
            test_confirmed_cases, 'bo')

    plt.grid()
    plt.legend(loc='best')
    plt.show()

def logit_fit(x, train_confirmed_cases, test_confirmed_cases, n_future_days):
    fig, ax, = plotData(x, train_confirmed_cases, False, 1.5)

    # logistic fit on data
    logit_model = LogisticModel(normalize=True)
    fitted, _ = logit_model.fit(x, train_confirmed_cases)
    ax.plot(x, fitted, 'g-', label='logistic fit')

    # extrapolation
    _x, preds = logit_model.predict(n_future_days)
    ax.plot(_x, preds, 'g--', label='predictions')
    ax.plot(np.arange(len(train_confirmed_cases), 
                      len(train_confirmed_cases)+len(test_confirmed_cases), 1),
            test_confirmed_cases, 'bo')

    plt.grid()
    plt.legend(loc='best')
    plt.show()

def new_cases_plot(n_avg, confirmed_cases):
    confirmed_averaged = moving_average(confirmed_cases, n=n_avg)

    new_cases = []
    for epoch, csp in enumerate(confirmed_averaged):
        if epoch==0:
            new_cases.append(csp)
        else: 
            new_cases.append(confirmed_averaged[epoch] - confirmed_averaged[epoch-1])
    new_cases = np.array(new_cases)

    fig = plt.figure(figsize=figsize(1.5,1))
    ax = fig.add_subplot(111)
    ax.plot(new_cases, 'o')
    ax.set_xlabel('days')
    ax.set_ylabel('new cases day-to-day')
    plt.title(f'on {n_avg}-day average confirmed cases')

    plt.grid()
    plt.show()

def averaged_new_cases_v_total_cases(confirmed_cases, period):
    # new cases for every period of days
    iterations, *_ = confirmed_cases.shape
    confirmed_cases_periodically = []
    epoch = 0
    while epoch < iterations:
        confirmed_cases_periodically.append(confirmed_cases[epoch])
        epoch += period

    confirmed_cases_periodically = np.array(confirmed_cases_periodically)

    # new cases every period of days
    new_cases = []
    for epoch, csp in enumerate(confirmed_cases_periodically):
        if epoch==0:
            new_cases.append(csp)
        else: 
            new_cases.append(confirmed_cases_periodically[epoch] - confirmed_cases_periodically[epoch-1])

    new_cases = np.array(new_cases)

    fig = plt.figure(figsize=figsize(1.5,1))
    ax = fig.add_subplot(111)
    ax.loglog(confirmed_cases_periodically, new_cases, '-', label='data')
    plt.xlabel(f'total confirmed cases every {period} days')
    plt.ylabel(f'new cases')

    # fitting it linearly to check if the growth is exponential
    _x = np.linspace(0, np.max(confirmed_cases_periodically))
    k, b = np.polyfit(np.log(confirmed_cases_periodically), np.log(new_cases), 1)
    y = _x**k * np.exp(b)
    ax.plot(_x, y, 'r--', label='loglinear fit')

    # exponential growth ground
    ax.plot(_x, _x, 'k:', label='exp growth')

    plt.grid()
    plt.legend()
    plt.show()

def gaussian_processes_extrapolation(x, confirmed_cases, train_confirmed_cases, test_confirmed_cases):
    # normalize the data
    normalized_train_confirmed_cases = normalize(train_confirmed_cases)

    # kernel assemble
    kernel = ConstantKernel() + Matern(nu=1.5) + WhiteKernel(noise_level=1)
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(x.reshape(-1,1), normalized_train_confirmed_cases.reshape(-1,1))

    # simulated data
    n_future_days_list = [10, 50, 100, 365]

    for i, n_d in enumerate(n_future_days_list):
        x_pred = np.arange(0, len(x)+n_d-1).reshape(-1,1)
        y_pred, sigma = gp.predict(x_pred, return_std=True)
        x_pred = x_pred.ravel()
        y_pred, sigma = y_pred.ravel(), sigma.ravel()

        # restore data
        y_pred = restore(y_pred, train_confirmed_cases)
        sigma = restore(sigma, train_confirmed_cases)
        y_pred_lower_bound = y_pred - sigma
        y_pred_upper_bound = y_pred + sigma

        fig, ax = plotData(x, train_confirmed_cases, False, 1.5, 1)
        ax.plot(x_pred, y_pred, label='Gaussian Processes - expected fit')
        ax.plot(x_pred, y_pred_lower_bound, color='red', label='95\% confidence interval')
        ax.plot(x_pred, y_pred_upper_bound, color='red')
        ax.fill_between(x_pred, 
                        y_pred_lower_bound, y_pred_upper_bound, 
                        color='red', alpha='0.3')
        
        ax.plot(np.arange(len(train_confirmed_cases), 
                          len(train_confirmed_cases)+len(test_confirmed_cases)), 
                test_confirmed_cases, 'bo', linestyle='None')
        plt.title(f'R2 score: {np.round(r2_score(confirmed_cases, y_pred[:len(confirmed_cases)]),4)}')
        plt.grid()
        plt.legend(loc='best')
        plt.show()

def sir_model(S0, I0, R0, confirmed_cases, recovered_cases, split_ratio):
    train_confirmed_cases, test_confirmed_cases = train_test_split(confirmed_cases, split_ratio)
    train_recovered_cases, test_recovered_cases = train_test_split(recovered_cases, split_ratio)
    
    initial_conditions = [S0, I0, R0]
    n_future_days = len(train_confirmed_cases) * 5

    sir_model = SIR()
    beta, gamma = sir_model.fit(train_confirmed_cases, train_recovered_cases, initial_conditions)
    print(beta, gamma)
    sol = sir_model.predict(n_future_days)

    epidemics_start = dt.datetime(2020, 2, 25)
    end = epidemics_start + dt.timedelta(days=n_future_days)
    days = mdates.drange(epidemics_start, end, dt.timedelta(days=1))

    fig = plt.figure(figsize=figsize(1.5, 1))
    ax = fig.add_subplot(111)
    _ = fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    _ = fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))

    ax.plot(days, sol.y[0], 'k-', label='S(t)')
    ax.plot(days, sol.y[1], 'r-', label='I(t)')
    ax.plot(days, sol.y[2], 'g-', label='R(t)')
    ax.plot(days[:len(confirmed_cases)],
            confirmed_cases, 
            linestyle='None', marker='x', color='red', alpha=0.7, 
            label='train confirmed cases')
    ax.plot(days[len(confirmed_cases):len(confirmed_cases)+len(test_confirmed_cases)], 
            test_confirmed_cases, 
            'ro', linestyle='None', 
            label='test confirmed cases')
    ax.plot(days[:len(recovered_cases)],
            recovered_cases, 
            linestyle='None', marker='x', color='green', alpha=0.7, 
            label='train recovered cases')
    ax.plot(days[len(recovered_cases):len(recovered_cases)+len(test_recovered_cases)], 
            test_recovered_cases, 
            'go', linestyle='None', 
            label='test recovered cases')

    _ = plt.gcf().autofmt_xdate()
    plt.ylabel('number of people')

    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


def main():
    latexconfig()

    # data
    confirmed_cases = np.loadtxt('data/confirmed_cases.dat')
    recovered_cases = np.loadtxt('data/recovered_cases.dat')

    # ratio = 1
    # train_confirmed_cases, test_confirmed_cases = train_test_split(confirmed_cases, ratio)
    # train_recovered_cases, test_recovered_cases = train_test_split(recovered_cases, ratio)
    
    # # days since first case
    # x = np.arange(len(train_confirmed_cases))

    # susceptible-infectious-recovered model
    S0 = 2200
    I0 = confirmed_cases[0]
    R0 = 0
    sir_model(S0, I0, R0, confirmed_cases, recovered_cases, split_ratio=1)
    

if __name__ == "__main__":
    main()