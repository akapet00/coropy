import datetime as dt

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from sklearn.metrics import r2_score

from covid_19.utils import normalize, restore, moving_average, train_test_split
from covid_19.plotting import plotData, figsize
from covid_19.compartmental_models import SEIR


def seir_sim(S0, E0, I0, R0, confirmed_cases, recovered_cases, split_ratio, epidemics_start_date):
    train_confirmed_cases, test_confirmed_cases = train_test_split(confirmed_cases, split_ratio)
    train_recovered_cases, test_recovered_cases = train_test_split(recovered_cases, split_ratio)
    
    initial_conditions = [S0, E0, I0, R0]
    n_future_days = len(train_confirmed_cases) * 3

    seir_model = SEIR()
    beta, delta, alpha, gamma = seir_model.fit(train_confirmed_cases, train_recovered_cases, initial_conditions)
    R0 = beta/(alpha+gamma)
    sol = seir_model.predict(n_future_days)

    end = epidemics_start_date + dt.timedelta(days=n_future_days)    
    days = mdates.drange(epidemics_start_date, end, dt.timedelta(days=1))

    fig = plt.figure(figsize=figsize(1, 1))
    ax = fig.add_subplot(111)
    _ = fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    _ = fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))

    ax.plot(days, sol.y[0], 'k-', label='$S(t)$')
    ax.plot(days, sol.y[1], 'b-', label='$E(t)$')
    ax.plot(days, sol.y[2], 'r-', label='$I(t)$')
    ax.plot(days, sol.y[3], 'g-', label='$R(t)$')
    ax.plot(days[:len(train_confirmed_cases)],
            train_confirmed_cases, 
            linestyle='None', marker='o', color='red', alpha=0.7, 
            label='infected data')
    ax.plot(days[:len(train_recovered_cases)],
            train_recovered_cases, 
            linestyle='None', marker='o', color='green', alpha=0.7, 
            label='removed data')

    if test_confirmed_cases.size > 0:
        ax.scatter(days[len(train_confirmed_cases):len(train_confirmed_cases)+len(test_confirmed_cases)],
                test_confirmed_cases,
                facecolors='none', edgecolors='r', linestyle='None',
                label='test infected')

        ax.scatter(days[len(train_recovered_cases):len(train_recovered_cases)+len(test_recovered_cases)],
                test_recovered_cases,
                facecolors='none', edgecolors='g', linestyle='None',
                label='test removed')
        plt.axis([mdates.date2num(epidemics_start_date - dt.timedelta(days=1)), days[confirmed_cases.size + 1], -100, 1750])
        _ = fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))

        plt.legend(loc='best')
    else:
        plt.legend(loc='lower right')

    _ = plt.gcf().autofmt_xdate()
    plt.ylabel('$N$')

    plt.grid()
    plt.show()
    return R0

# full raw data
confirmed_cases = np.loadtxt('data/confirmed_cases.dat')
recovered_cases = np.loadtxt('data/recovered_cases.dat')
death_cases = np.loadtxt('data/death_cases.dat')
removed_cases = recovered_cases + death_cases
active_cases = confirmed_cases - removed_cases

############ 
# 1st wave #
############
start_date_1 = dt.datetime(2020, 2, 25)
end_date_1 = dt.datetime(2020, 6, 4)
diff = abs((end_date_1 - start_date_1).days)
removed_cases_1 = removed_cases[:diff+1]
active_cases_1 = active_cases[:diff+1]

#split_ratio = 1
#R0 = seir_sim(
#    S0=2500,
#    E0=0,
#    I0=active_cases_1[0],
#    R0=removed_cases_1[0],
#    confirmed_cases=active_cases_1,
#    recovered_cases=removed_cases_1,
#    split_ratio=split_ratio,
#    epidemics_start_date=start_date_1,
#    )

############ 
# 2nd wave #
############
start_date_2 = dt.datetime(2020, 6, 5)
removed_cases_2 = removed_cases[diff+1:] - removed_cases[diff+1]
active_cases_2 = active_cases[diff+1:]

split_ratio = 1.
R0 = seir_sim(
    S0=4000,
    E0=0,
    I0=active_cases_2[0],
    R0=removed_cases_2[0],
    confirmed_cases=active_cases_2,
    recovered_cases=removed_cases_2,
    split_ratio=split_ratio,
    epidemics_start_date=start_date_2,
    )
