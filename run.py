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
from covid_19.compartmental_models import SIR, SEIR

def exp_fit(x, train_confirmed_cases, test_confirmed_cases, n_future_days):
    fig, ax = plotData(x, train_confirmed_cases, False, 1.5)

    exp_model = ExponentialModel(normalize=False)
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
    fig.savefig(f'figs/exp-fit.pdf', bbox_inches='tight')
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
    fig.savefig(f'figs/logit-fit.pdf', bbox_inches='tight')
    plt.show()

def new_cases_plot(confirmed_cases, n_avg):
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
    ax.plot(new_cases, 'bo-')
    ax.set_xlabel('days')
    ax.set_ylabel(f'new cases avereged over {n_avg} days')

    plt.grid()
    fig.savefig(f'figs/{n_avg}-day-avg-conf-cases.pdf', bbox_inches='tight')
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
    ax.loglog(confirmed_cases_periodically, new_cases, 'b-', label='data')
    plt.xlabel(f'confirmed cases')
    plt.ylabel(f'new cases')

    # fitting it linearly to check if the growth is exponential
    _x = np.linspace(0, np.max(confirmed_cases_periodically))
    k, b = np.polyfit(np.log(confirmed_cases_periodically), np.log(new_cases), 1)
    y = _x**k * np.exp(b)
    ax.plot(_x, y, 'r--', label='loglinear fit')

    # exponential growth ground
    ax.plot(_x, _x, 'k:', label='exponential growth')

    plt.grid()
    plt.legend()
    fig.savefig(f'figs/new-v-total.pdf', bbox_inches='tight')
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

def sir_model(S0, I0, R0, confirmed_cases, recovered_cases, split_ratio, epidemics_start_date):
    train_confirmed_cases, test_confirmed_cases = train_test_split(confirmed_cases, split_ratio)
    train_recovered_cases, test_recovered_cases = train_test_split(recovered_cases, split_ratio)
    
    initial_conditions = [S0, I0, R0]
    n_future_days = len(train_confirmed_cases) * 5

    sir_model = SIR()
    beta, gamma = sir_model.fit(train_confirmed_cases, train_recovered_cases, initial_conditions)
    print(beta, gamma)
    sol = sir_model.predict(n_future_days)

    end = epidemics_start_date + dt.timedelta(days=n_future_days)    
    days = mdates.drange(epidemics_start_date, end, dt.timedelta(days=1))

    fig = plt.figure(figsize=figsize(1.5, 1))
    ax = fig.add_subplot(111)
    _ = fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    _ = fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))

    #ax.plot(days, sol.y[0], 'k-', label='susceptible')
    ax.plot(days, sol.y[1], 'r-', label='infected')
    ax.plot(days, sol.y[2], 'g-', label='recovered/diceased')
    ax.plot(days[:len(train_confirmed_cases)],
            train_confirmed_cases, 
            linestyle='None', marker='x', color='red', alpha=0.7, 
            label='train confirmed cases')
    ax.plot(days[len(train_confirmed_cases):len(train_confirmed_cases)+len(test_confirmed_cases)], 
            test_confirmed_cases, 
            'ro', linestyle='None', 
            label='test confirmed cases')
    ax.plot(days[:len(train_recovered_cases)],
            train_recovered_cases, 
            linestyle='None', marker='x', color='green', alpha=0.7, 
            label='train recovered cases')
    ax.plot(days[len(train_recovered_cases):len(train_recovered_cases)+len(test_recovered_cases)], 
            test_recovered_cases, 
            'go', linestyle='None', 
            label='test recovered cases')

    _ = plt.gcf().autofmt_xdate()
    plt.ylabel('number of people')

    plt.grid()
    plt.legend(loc='upper right')
    fig.savefig(f'figs/sir.pdf', bbox_inches='tight')
    plt.show()

def seir_model(S0, E0, I0, R0, confirmed_cases, recovered_cases, split_ratio, epidemics_start_date):
    train_confirmed_cases, test_confirmed_cases = train_test_split(confirmed_cases, split_ratio)
    train_recovered_cases, test_recovered_cases = train_test_split(recovered_cases, split_ratio)
    
    initial_conditions = [S0, E0, I0, R0]
    n_future_days = len(train_confirmed_cases) * 3

    seir_model = SEIR()
    beta, delta, alpha, gamma = seir_model.fit(train_confirmed_cases, train_recovered_cases, initial_conditions)
    R0 = beta/(delta+gamma)
    sol = seir_model.predict(n_future_days)

    end = epidemics_start_date + dt.timedelta(days=n_future_days)    
    days = mdates.drange(epidemics_start_date, end, dt.timedelta(days=1))

    fig = plt.figure(figsize=figsize(1.5, 1))
    ax = fig.add_subplot(111)
    _ = fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    _ = fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))

    #ax.plot(days, sol.y[0], 'k-', label='susceptible')
    #ax.plot(days, sol.y[1], 'b-', label='exposed')
    ax.plot(days, sol.y[2], 'r-', label='infected')
    ax.plot(days, sol.y[3], 'g-', label='recovered/diceased')
    ax.plot(days[:len(train_confirmed_cases)],
            train_confirmed_cases, 
            linestyle='None', marker='x', color='red', alpha=0.7, 
            label='fitted confirmed')
    ax.plot(days[len(train_confirmed_cases):len(train_confirmed_cases)+len(test_confirmed_cases)], 
            test_confirmed_cases, 
            'ro', linestyle='None', 
            label='test confirmed')
    ax.plot(days[:len(train_recovered_cases)],
            train_recovered_cases, 
            linestyle='None', marker='x', color='green', alpha=0.7, 
            label='fitted removed')
    ax.plot(days[len(train_recovered_cases):len(train_recovered_cases)+len(test_recovered_cases)], 
            test_recovered_cases, 
            'go', linestyle='None', 
            label='test removed')

    _ = plt.gcf().autofmt_xdate()
    plt.ylabel('number of people')

    plt.grid()
    plt.legend(loc='upper right')
    fig.savefig(f'figs/seir-{split_ratio}-split-ratio.pdf', bbox_inches='tight')
    plt.show()
    return R0

def main():
    latexconfig()

    # cro data
    start_date = dt.datetime(2020, 2, 25)
    confirmed_cases = np.loadtxt('data/cro/confirmed_cases.dat')
    recovered_cases = np.loadtxt('data/cro/recovered_cases.dat')
    death_cases = np.loadtxt('data/cro/death_cases.dat')
    removed_cases = recovered_cases + death_cases
    
    # ratio = 1
    # train_confirmed_cases, test_confirmed_cases = train_test_split(confirmed_cases, ratio)
    # train_removed_cases, test_removed_cases = train_test_split(removed_cases, ratio)
    
    # # days since first case
    # x = np.arange(len(train_confirmed_cases))

    # # exp fit
    # exp_fit(x, train_confirmed_cases, test_confirmed_cases, n_future_days=7)

    # # logit fit 
    # logit_fit(x, train_confirmed_cases, test_confirmed_cases, n_future_days=7)

    # # new cases averaged 
    # new_cases_plot(confirmed_cases, n_avg=7)

    # # new cases v total cases averaged
    # averaged_new_cases_v_total_cases(confirmed_cases, period=7)

    # susceptible-exposed-infected-recovered model
    split_ratio = [0.88, 0.96, 1.0]
    R0 = np.empty(shape=(2, len(split_ratio)))
    R0[0, :] = np.array(split_ratio)
    for i, ratio in enumerate(split_ratio):
        R0[1, i] = (seir_model(S0=5000, 
                               E0=0, 
                               I0=confirmed_cases[0], 
                               R0=recovered_cases[0], 
                               confirmed_cases=confirmed_cases, 
                               recovered_cases=recovered_cases, 
                               split_ratio=ratio, 
                               epidemics_start_date=start_date))

    file_name = f'params/{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'
    np.savetxt(file_name, R0)
    
if __name__ == "__main__":
    main()