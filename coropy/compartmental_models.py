import datetime as dt
import logging
import sys
import time
import warnings

import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from .utils import (mse, rmse, msle, mae)
from .utils import (normalize, restore)


__all__ = ['SEIRModel', 'SEIRDModel']


def _SEIR(t, y, beta, delta, alpha, gamma):
    """Return the SEIR compartmental system values. For details check: 
    https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology
    
    Parameters
    ----------
    t : numpy.ndarray
        Discrete time points.
    y : list or tuple
        Values of S, E, I and R.
    beta : float
        Transition (infectious) rate controls the rate of spread. 
    delta : float
        Direct transition rate between S and E individual. 
    alpha : float
        Incubation rate, the reciprocal value of the incubation period.
    gamma : float
        Recovery (or mortality) rate.
        
    Returns
    -------
    list
        Values of the SEIR compartmental model free parameters.
    """
    S, E, I, R = y
    N = S + E + I + R
    return [
        -beta*S*I/N - delta*S*E, 
        beta*S*I/N - alpha*E + delta*S*E, 
        alpha*E - gamma*I, 
        gamma*I,
    ]

def _SEIRD(t, y, beta, alpha, gamma, mu):
    """Return the SEIRD compartmental system values. For details check:
    https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology
    
    Parameters
    ----------
    t : numpy.ndarray
        Discrete time points.
    y : list or tuple
        Values of S, E, I and R.
    beta : float
        Transition (infectious) rate controls the rate of spread. 
    alpha : float
        Incubation rate, the reciprocal value of the incubation period.
    gamma : float
        Recovery rate.
    mu : float
        Mortality rate.
        
    Returns
    -------
    list
        Values of the SEIRD compartmental model free parameters.
    """
    S, E, I, R, D = y
    N = S + E + I + R + D
    return [
        -beta*S*I/N, 
        beta*S*I/N - alpha*E, 
        alpha*E - gamma*I - mu*I, 
        gamma*I,
        mu*I,
    ]


class CompartmentalModel(object):
    """General compartmental model class."""

    def __init__(self, loss_fun='mse', calc_ci=False, **kwargs):
        """Constructor.
        
        Parameters
        ----------
        loss_fun: str, optional
            Estimator that measures distance between true and predicted
            values. Loss function is set to `mean_squared_error`.
        calc_ci: bool, optional
            Calculate confidence interval by providing additional
            keyword arguments specified below.
        kwargs : dict, optional
            If `calc_ci` is set to True, and the following keyword
            arguments are specified, `CompartmenalModel.simulate` will
            have its output bounded in confidence intervals. Keyword
            are as follows: `pcr_sens` (float), measure of the
            proportion of positives that are correctly identified (e.g.
            the percentage of sick people who are correctly identified
            as being infected), `pcr_spec` (float), the measure of the
            proportion of negatives that are correctly identified (e.g.
            the percentage of healthy people who are correctly
            identified as not infected), `daily_tests` (numpy.ndarray),
            time series of new daily completed tests.
        """
        if loss_fun in ['mean_squared_error', 'mse', 'MSE']:
            self.loss_fun = mse
        elif loss_fun in ['root_mean_squared_error', 'rmse', 'RMSE']:
            self.loss_fun = rmse
        elif loss_fun in ['mean_squared_logarithmic_error', 'msle', 'MSLE']:
            self.loss_fun = msle
        elif loss_fun in ['mean_absolute_error', 'mae', 'MAE']:
            self.loss_fun = mae
        else:
            raise ValueError('Given loss function is not supported.')
        
        self.calc_ci = calc_ci
        if self.calc_ci:
            if kwargs:
                for kw in kwargs.keys():
                    if kw == 'pcr_sens':
                        assert isinstance(kwargs[kw], (float, )), \
                            '`pcr_sens` has to be a floating point number.'
                        self.pcr_sens = kwargs[kw]
                    elif kw == 'pcr_spec':
                        assert isinstance(kwargs[kw], (float, )), \
                            '`pcr_spec` has to be a floating point number.'
                        self.pcr_spec = kwargs[kw]
                    elif kw == 'daily_tests':
                        assert isinstance(kwargs[kw], np.ndarray), \
                            '`daily_test` has to be of `numpy.ndarray` type.'
                        self.daily_tests = kwargs[kw]
                    else:
                        raise KeyError('Optional keyword argument invalid.')
            else:
                self.calc_ci = False
                warnings.warn(
                    'Confidence interval will not be calculated'
                    'Optional `kwargs` are not specified.')
        self.params = None
    
    @staticmethod
    def calculate_ci(pcr_sens, pcr_spec, daily_positive, cum_removed,
        daily_tests):
        """Return two arrays: the lower confidence interval bound, the
        second row is the upper confidence interval bound.

        Parameters
        ----------
        pcr_sens : float or numpy.ndarray
            PCR-test sensitivity, the measure of the proportion of
            `daily_positives` that are correctly classified.
        pcr_spec : float or numpy.ndarray
            PCR-test pecificity, the measure of the proportion of
            the correctly classified negative tests.
        daily_positive : numpy.ndarray
            Daily number of newly confirmed positive infections.
        cum_removed : numpy.ndarray
            Cumulative number of confirmed recoveries and deaths.
        daily_tests : numpy.ndarray
            Daily number of tests.

        Returns
        -------
        tuple
            Tuple with 4 arrays: lower 95% CI bound, lower CI bound,
            upper CI bound and upper 95% CI bound.
        """
        # LOWER BOUND
        # lower bound CI
        tp_lb = pcr_sens * daily_positive
        active_lb = np.cumsum(tp_lb) - cum_removed
        active_lb[np.where(active_lb<0)] = 0
        # lower bound 95% CI
        std_sens_err = np.sqrt(np.divide(
            (1 - pcr_sens) * pcr_sens, 
            daily_positive, 
            out=np.zeros(daily_positive.shape, dtype=float), 
            where=daily_positive!=0,))
        sens_lb_ci = pcr_sens - 1.96 * std_sens_err
        tp_lb_ci = sens_lb_ci * daily_positive
        active_lb_ci = np.cumsum(tp_lb_ci) - cum_removed
        active_lb_ci[np.where(active_lb_ci<0)] = 0 
        # UPPER BOUND
        # yesterday's test gives today's result
        daily_tests = np.concatenate((np.array([daily_tests[0]]),
            daily_tests[:-1]))
        daily_negative = daily_tests - daily_positive
        # upper bound CI
        tn = pcr_spec * daily_negative
        fn = daily_negative - tn
        tp_ub = fn + daily_positive
        active_ub = np.cumsum(tp_ub) - cum_removed
        # upper bound 95% CI
        std_spec_err = np.sqrt(np.divide(
            (1 - pcr_spec) * pcr_spec,
            daily_negative,
            out=np.zeros(daily_negative.shape, dtype=float), 
            where=daily_negative!=0,))
        spec_ub_ci = pcr_spec - 1.96 * std_spec_err
        tn_ci = spec_ub_ci * daily_negative
        fn_ci = daily_negative - tn_ci
        tp_ub_ci = fn_ci + daily_positive
        active_ub_ci = np.cumsum(tp_ub_ci) - cum_removed
        return (active_lb_ci, active_lb, active_ub, active_ub_ci)
     

class SEIRModel(CompartmentalModel):
    """SEIR model class."""

    def fit(self, cum_positives, cum_recovered, cum_deceased, IC,
        guess=[0.1, 0.1, 0.1, 0.1]):
        """Fit SEIR model.
        
        Parameters
        ----------
        cum_positives : numpy.ndarray
            Cumaltive number of confirmed positive infections.
        cum_recovered : numpy.ndarray
            Cumulative number of confirmed recoveries. 
        cum_deceased : numpy.ndarray
            Cumulative number of deaths.
        IC : list
            Initial values of S, E, I and R at the first day.
        guess : list, optional
            Initial guess for parameters to be fitted.
        
        Returns
        -------
        tuple
            Fitted epidemiological parameters.
        list
            Loss values during the optimization procedure.
        """
        assert isinstance(cum_positives, np.ndarray), \
            '`cum_positives` has to be of `numpy.ndarray` type.'
        assert isinstance(cum_recovered, np.ndarray), \
            '`cum_recovered` has to be of `numpy.ndarray` type.'
        assert isinstance(cum_deceased, np.ndarray), \
            '`cum_deceased` has to be of `numpy.ndarray` type.'
        self.daily_positive = np.concatenate((
            np.array([cum_positives[0]]), np.diff(cum_positives)))
        self.active = cum_positives - cum_recovered - cum_deceased
        self.removed = cum_recovered + cum_deceased
        self.IC = IC
        loss_values = []
        def _print_loss(p):
            """Optimizer callback."""
            loss_values.append(
                SEIRModel._loss(p, self.active, self.removed, self.IC, 
                    self.loss_fun))

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.info(f'L-BFGS-B optimization started: {dt.datetime.now()}')
        start_stopwatch = time.time()
        opt = minimize(
            fun=SEIRModel._loss, 
            x0=guess,
            args=(self.active, self.removed, self.IC, self.loss_fun),
            method='L-BFGS-B',
            bounds=[(1e-5, 1.0)] * len(guess),
            options={'maxiter': 1000, 'disp': True},
            callback=_print_loss,
            )
        elapsed = time.time() - start_stopwatch
        logging.info(f'Elapsed time: {round(elapsed, 4)}s')
        self.params = opt.x
        return self.params, loss_values

    def simulate(self):
        """Simulate S, E, I and R based on the fitted epidemiological
        parameters.
            
        Returns
        -------
        tuple
            S, E, I and R simulated values.
        """
        sol = solve_ivp(
            fun=_SEIR, 
            t_span=(0, self.active.size), 
            y0=self.IC, 
            args=tuple(self.params),
            method='RK45', 
            t_eval=np.arange(0, self.active.size, 1), 
            vectorized=True,
            )
        if self.calc_ci:
            (active_lb_ci, active_lb, active_ub, active_ub_ci) = \
                self.calculate_ci(self.pcr_sens, self.pcr_spec,
                    self.daily_positive, self.removed, self.daily_tests)
            I_ci = np.r_[
                active_lb_ci.reshape(1, -1),
                active_lb.reshape(1, -1),
                sol.y[2].reshape(1, -1),
                active_ub.reshape(1, -1),
                active_ub_ci.reshape(1, -1)]
            return (sol.y[0], sol.y[1], I_ci, sol.y[3])
        return (sol.y[0], sol.y[1], sol.y[2], sol.y[3])

    def forecast(self, n_days):
        """Predict S, E, I and R based on the fitted epidemiological
        parameters.
        
        Parameters
        ----------
        n_days : int
            Number of days to forecast S, E, I and R in the future.

        Returns
        -------
        tuple
            S, E, I and R predicted values.
        """
        assert isinstance(n_days, (int, )), '`n_days` must be an integer.'
        eff_idx = self.active.size
        sol = solve_ivp(
            fun=_SEIR, 
            t_span=(0, eff_idx + n_days),
            y0=self.IC,
            args=tuple(self.params),
            method='RK45',
            t_eval=np.arange(0, eff_idx + n_days, 1), 
            vectorized=True,
            )
        return (sol.y[0][eff_idx:], sol.y[1][eff_idx:], sol.y[2][eff_idx:],
            sol.y[3][eff_idx:],)
    
    def __str__(self):
        return('SEIR model class')

    def __repr__(self):
        return self.__str__()

    @property
    def get_params(self):
        if self.params is None:
            raise ValueError('No fitted parameters. Call `fit` method first.')
        return (self.params)
    
    @staticmethod
    def _loss(params, active, cum_removed, IC, loss_fun):
        """Calculate and return the loss function between the actual
        and predicted values.
        
        Parameters
        ----------
        params : list
            Epidemiological parameters.
        active: numpy.ndarray
            Time series of currently active infected individuals.
        cum_removed : numpy.ndarray
            Cumulative number of confirmed recoveries and deaths.
        IC: list
            Initial conditions
        loss_fun : str
            Estimator.
                
        Returns
        -------
        float
            Loss between the actual and predicted value of confirmed
            and recovered individuals.
        """
        size = active.size
        sol = solve_ivp(
            fun=_SEIR, 
            t_span=(0, size), 
            y0=IC, 
            args=params,
            method='RK45', 
            t_eval=np.arange(0, size, 1), 
            vectorized=True,
        )
        return loss_fun(sol.y[2], active)\
               + loss_fun(sol.y[3], cum_removed)


class SEIRDModel(CompartmentalModel):
    """SEIRD model class."""

    def fit(
        self, 
        active_cases, 
        recovered_cases, 
        death_cases, 
        initial_conditions, 
        initial_guess=[0.5, 0.1, 0.01, 0.01],
        ):
        """Fit SEIRD model.
        
        Parameters
        ----------
        active_cases : numpy.ndarray
            Time series of currently active infected individuals.
        recovered_cases : numpy.ndarray
            Time series of recovered individuals.
        death_cases : numpy.ndarray
            Time series of deceased individuals.
        initial_conditions : list
            Values of S, E, I, R and D at the first day.
        initial_guess : list, optional
            Array of real elements by means of possible values of
            independent variables.
        
        Returns
        -------
        tuple
            Fitted epidemiological parameters: beta, alpha, gamma and
            mu rate.
        list
            Loss values during the optimization procedure.
        """
        self.active_cases = active_cases
        self.recovered_cases = recovered_cases
        self.death_cases = death_cases

        loss = []
        def print_loss(p):
            """Optimizer callback."""
            loss.append(
                SEIRDModel._loss(
                    p,
                    self.active_cases,
                    self.recovered_cases,
                    self.death_cases,
                    initial_conditions,
                    self.loss_fn
                    )
            )
            
        self.y0 = initial_conditions
        opt = minimize(
            fun=SEIRDModel._loss, 
            x0=initial_guess,
            args=(
                self.active_cases,
                self.recovered_cases,
                self.death_cases,
                self.y0,
                self.loss_fn,
                ),
            method='L-BFGS-B',
            bounds=[(0, 1), (0, 1), (0, 1), (0, 1),],
            options={'disp': True, 'maxiter': 1000},
            callback=print_loss,
        )
        self.beta, self.alpha, self.gamma, self.mu = opt.x
        return (self.beta, self.alpha, self.gamma, self.mu), loss

    def simulate(self):
        """Simulate S, E, I, R and D based on the fitted
        epidemiological parameters.
            
        Returns
        -------
        tuple
            S, E, I, R and D simulated values.
        """
        sol = solve_ivp(
            fun=_SEIRD, 
            t_span=(0, self.active_cases.size), 
            y0=self.y0, 
            args=(self.beta, self.alpha, self.gamma, self.mu),
            method='RK45', 
            t_eval=np.arange(0, self.active_cases.size, 1), 
            vectorized=True,
        )
        if (self.sensitivity and self.specificity and 
                 self.total_tests is not None):
            (lower_bound, upper_bound) = self.calculate_ci(
                self.sensitivity, 
                self.specificity,
                self.new_positives,
                self.active_cases,
                self.recovered_cases,
                self.total_tests,
                )
            I_ci = np.r_[
                lower_bound.reshape(1, -1),
                sol.y[2].reshape(1, -1),
                upper_bound.reshape(1, -1),
            ]
            return (sol.y[0], sol.y[1], I_ci, sol.y[3], sol.y[4])
        return (sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4])
    
    def forecast(self, n_days):
        """Predict S, E, I, R and D based on the fitted
        epidemiological parameters.
        
        Parameters
        ----------
        n_days : int
            Number of days to forecast S, E, I, R and D in the future.

        Returns
        -------
        tuple
            S, E, I, R and D predicted values.
        """
        assert isinstance(n_days, (int, )), '`n_days` must be an integer.'
        eff_idx = self.active_cases.size
        sol = solve_ivp(
            fun=_SEIRD, 
            t_span=(0, eff_idx + n_days),
            y0=self.y0,
            args=(self.beta, self.alpha, self.gamma, self.mu),
            method='RK45',
            t_eval=np.arange(0, eff_idx + n_days, 1), 
            vectorized=True,
        )
        return (
            sol.y[0][eff_idx:], 
            sol.y[1][eff_idx:], 
            sol.y[2][eff_idx:], 
            sol.y[3][eff_idx:], 
            sol.y[4][eff_idx:],
        )

    @staticmethod
    def _loss(
        params,
        active_cases,
        recovered_cases,
        death_cases,
        initial_conditions,
        loss_fn,
        ):
        """Calculate and return the loss function between actual and
        predicted values.
        
        Parameters
        ----------
        params : list
            Values of beta, alpha, gamma and mu rates.
        active_cases : numpy.ndarray
            Time series of currently active infected individuals.
        recovered_cases : numpy.ndarray
            Time series of recovered individuals.
        death_cases : numpy.ndarray
            Time series of deceased individuals.
        initial_conditions : list
            Values of S, E, I, R and D at the first day.
        loss_fn : str, optional
            Loss function is `mse` by default, choose between `mse`,
            `rmse` and `msle`.
        
        Returns
        -------
        float
            Loss between the actual and predicted value of confirmed,
            recovered and deceased individuals.
        """
        size = active_cases.size
        sol = solve_ivp(
            fun=_SEIRD, 
            t_span=(0, size), 
            y0=initial_conditions, 
            args=params,
            method='RK45', 
            t_eval=np.arange(0, size, 1), 
            vectorized=True,
        )
        return loss_fn(sol.y[2], active_cases) \
               + loss_fn(sol.y[3], recovered_cases) \
               + loss_fn(sol.y[4], death_cases)