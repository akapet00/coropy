import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score

from SIR.utils import normalize, restore, moving_average
from SIR.plotting import plotData, figsize, latexconfig
from SIR.model import ExponentialModel, LogisticModel
from SIR.SIR import SIRModel

def main():
    latexconfig()

    # data
    confirmed_cases = np.loadtxt('data/confirmed_cases.dat')
    recovered_cases = np.loadtxt('data/recovered_cases.dat')

    # train-test split 
    ratio = 0.96
    train_size = int(ratio * len(confirmed_cases))

    train_confirmed_cases = confirmed_cases[:train_size]
    test_confirmed_cases = confirmed_cases[train_size:]
    
    train_recovered_cases = recovered_cases[:train_size]
    test_recovered_cases = recovered_cases[train_size:]
    
    # days since first case
    x = np.arange(len(train_confirmed_cases))

    ##########
    # log fit #
    ########## 
    fig, ax = plotData(x, train_confirmed_cases, False, 1.5)

    exp_model = ExponentialModel()
    fitted, _ = exp_model.fit(x, train_confirmed_cases)
    ax.plot(x, fitted, 'r-', label='exponential fit')

    # extrapolation 
    n_future_days = 7
    _x, preds = exp_model.predict(n_future_days)
    ax.plot(_x, preds, 'r--', label='predictions')
    ax.plot(np.arange(len(train_confirmed_cases), 
                      len(train_confirmed_cases)+len(test_confirmed_cases), 1),
            test_confirmed_cases, 'bo')

    plt.grid()
    plt.legend(loc='best')
    plt.show()

    #############
    # logit fit #
    #############
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

    ######################################################
    # new cases over time, avereged to some period n_avg #
    ######################################################
    n_avg = 7
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

    #############################################################
    # new confirmed cases v. total confirmed cases every n days #
    #############################################################
    # log scaled: new confirmed cases vs total confirmed cases
    iterations, *_ = confirmed_cases.shape
    confirmed_cases_periodically = []
    epoch = 0
    period = 3
    while epoch < iterations:
        confirmed_cases_periodically.append(confirmed_cases[epoch])
        epoch += period

    confirmed_cases_periodically = np.array(confirmed_cases_periodically)

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
    ax.plot(_x, y, 'r--', label='linear fit')

    # exponential growth ground
    ax.plot(_x, _x, 'k:', label='exp growth')

    plt.grid()
    plt.legend()
    plt.show()

    #######################################################################
    # [just for the show] (over)fitting the data using Gaussian Processes #
    #######################################################################
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
    
    #############
    # SIR model #
    #############
    S0, I0, R0 = 2100, 2, 0 # rough estimate, from GP fit upper bound
    initial_conditions = [S0, I0, R0]
    n_future_days = len(train_confirmed_cases) * 5

    sir_model = SIRModel()
    beta, gamma = sir_model.fit(train_confirmed_cases, train_recovered_cases, initial_conditions)
    sol = sir_model.predict(n_future_days)

    fig = plt.figure(figsize=figsize(1.5, 1))
    ax = fig.add_subplot(111)
    ax.plot(sol.y[0], 'k-', label='S(t)')
    ax.plot(sol.y[1], 'r-', label='I(t)')
    ax.plot(sol.y[2], 'g-', label='R(t)')
    ax.plot(confirmed_cases, linestyle='None', marker='x', color='red', alpha=0.7, label='train confirmed cases')
    ax.plot(np.arange(len(train_confirmed_cases), 
                      len(train_confirmed_cases)+len(test_confirmed_cases)), 
            test_confirmed_cases, 'ro', linestyle='None', label='test confirmed cases')
    ax.plot(train_recovered_cases, linestyle='None', marker='x', color='green', alpha=0.7, label='train recovered cases')
    ax.plot(np.arange(len(train_recovered_cases), 
                      len(train_recovered_cases)+len(test_recovered_cases)), 
            test_recovered_cases, 'go', linestyle='None', label='test recovered cases')

    plt.xlabel('day')
    plt.ylabel('number of people')

    plt.grid()
    plt.legend(loc='upper right')
    plt.show()
if __name__ == "__main__":
    main()