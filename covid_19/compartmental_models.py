import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from covid_19.utils import (mse, rmse, msle, mae)
from covid_19.utils import normalize, restore


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

    def __init__(self, loss_fn='mse', sensitivity=None):
        """Constructor.
        
        Parameters
        ----------
        loss_fn: str, optional
            Loss function is `mse` by default
        sensitivity : float, optional
            Measure of the proportion of positives that are correctly
            identified (e.g., the percentage of sick people who are
            correctly identified as being infected). If the
            `sensitivity` is defined, the .95 confidence intervals will
            be caluclated.  
        """
        if loss_fn == 'mse':
            self.loss_fn = mse
        elif loss_fn == 'rmse':
            self.loss_fn = rmse
        elif loss_fn == 'msle':
            self.loss_fn = msle
        elif loss_fn == 'mae':
            self.loss_fn = mae
        else:
            raise ValueError('Given loss function is not supported.')
        if sensitivity:
            assert isinstance(sensitivity, (float, )), \
                'Sensitivity must be a floating point number in [0, 1]'
            self.sensitivity = sensitivity
        else:
            self.sensitivity = None
    
    @staticmethod
    def calculate_ci(sensitivity, fitted):
        """Return 2-d array with 3 rows, first row is the lower
        confidence interval bound, the second row is the fitted data
        and the last row is the upper confidence interval bound.

        Parameters
        ----------
        sensitivity : float or numpy.ndarray
            Test sensitivity. If the sensitivity is different in
            different intervals, it should be stored in the array-like
            format where the length is the same as the length of the
            data.

        fitted : numpy.ndarray
            Fitted data.
        """
        fitted_normalized = normalize(fitted)
        std_sensitivity_err = np.sqrt(np.divide(
            (1 - sensitivity) * sensitivity, 
            fitted_normalized, 
            out=np.zeros_like(fitted_normalized), 
            where=fitted_normalized!=0,
            )
        )
        sensitivity_ci = np.abs(1.960 * std_sensitivity_err)
        lower_bound = restore(
            (sensitivity - sensitivity_ci) * fitted_normalized, fitted
        )
        upper_bound = restore(
            (sensitivity + sensitivity_ci) * fitted_normalized, fitted
            )
        return np.r_[
            lower_bound.reshape(1, -1), 
            fitted.reshape(1, -1), 
            upper_bound.reshape(1, -1),
            ]
            

class SEIRModel(CompartmentalModel):
    """SEIR model class."""

    def fit(
        self, 
        active_cases, 
        removed_cases,
        initial_conditions,
        initial_guess=[0.001, 0.001, 0.001, 0.001],
        ):
        """Fit SEIR model.
        
        Parameters
        ----------
        active_cases : numpy.ndarray
            Time series of currently active infected individuals.
        removed_cases : numpy.ndarray
            Time series of recovered+deceased individuals.
        initial_conditions : list
            Values of S, E, I and R at the first day.
        initial_guess : list, optional
            Array of real elements by means of possible values of
            independent variables.
        
        Returns
        -------
        tuple
            Fitted epidemiological parameters: beta, delta, alpha and
            gamma rate.
        list
            Loss values during the optimization procedure.
        """
        self.active_cases = active_cases
        self.removed_cases = removed_cases
        loss = []
        def print_loss(p):
            """Optimizer callback."""
            loss.append(
                SEIRModel._loss(
                    p, 
                    self.active_cases, 
                    self.removed_cases, 
                    initial_conditions, 
                    self.loss_fn,
                    )
            )
            
        self.y0 = initial_conditions
        opt = minimize(
            fun=SEIRModel._loss, 
            x0=initial_guess,
            args=(self.active_cases, self.removed_cases, self.y0, self.loss_fn),
            method='L-BFGS-B',
            bounds=[(1e-5, 1.0), (1e-5, 1.0), (1e-5, 1.0), (1e-5, 1.0),],
            options={'maxiter': 1000, 'disp': True},
            callback=print_loss,
        )
        self.beta, self.delta, self.alpha, self.gamma = opt.x
        return (self.beta, self.delta, self.alpha, self.gamma), loss

    def predict(self, n_days):
        """Forecast S, E, I and R based on the fitted epidemiological
        parameters.
        
        Parameters
        ----------
        n_days : int
            Number of days for which the simulation will be performed.
            
        Returns
        -------
        tuple
            S, E, I and R simulated values.
        """
        sol = solve_ivp(
            fun=_SEIR, 
            t_span=(0, n_days), 
            y0=self.y0, 
            args=(self.beta, self.delta, self.alpha, self.gamma),
            method='RK45', 
            t_eval=np.arange(0, n_days, 1), 
            vectorized=True,
        )
        if self.sensitivity:
            I_ci = self.calculate_ci(self.sensitivity, sol.y[2])
            R_ci = self.calculate_ci(self.sensitivity, sol.y[3])
            return sol.y[0], sol.y[1], I_ci, R_ci
        return (sol.y[0], sol.y[1], sol.y[2], sol.y[3])
    
    @staticmethod
    def _loss(
        params, active_cases, removed_cases, initial_conditions, loss_fn
        ):
        """Calculate and return the loss function between actual and
        predicted values.
        
        Parameters
        ----------
        params : list
            Values of beta, delta, alpha and gamma rates.
        active_cases: numpy.ndarray
            Time series of currently active infected individuals.
        removed_cases: numpy.ndarray
            Time series of recovered+deceased individuals.
        initial_conditions: list
            Values of S, E, I and R at the first day.
        loss_fn : str, optional
            Loss function.
                
        Returns
        -------
        float
            Loss between the actual and predicted value of confirmed
            and recovered individuals.
        """
        size = active_cases.size
        sol = solve_ivp(
            fun=_SEIR, 
            t_span=(0, size), 
            y0=initial_conditions, 
            args=params,
            method='RK45', 
            t_eval=np.arange(0, size, 1), 
            vectorized=True,
        )
        return loss_fn(sol.y[2], active_cases) \
               + loss_fn(sol.y[3], removed_cases)


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
            bounds=[(1.e-6, 10.), (1.e-6, 10.), (1.e-6, 10.), (1.e-6, 10.),],
            options={
                'disp': True, 'maxiter': 1000, 'ftol': 1.e-21, 'gtol': 1.e-31
                },
            callback=print_loss,
        )
        self.beta, self.alpha, self.gamma, self.mu = opt.x
        return (self.beta, self.alpha, self.gamma, self.mu), loss

    def predict(self, n_days):
        """Forecast S, E, I and R based on the fitted epidemiological
        parameters.
        
        Parameters
        ----------
        n_days : int
            Number of days for which the simulation will be performed.
            
        Returns
        -------
        tuple
            S, E, I, R and D simulated values.
        """
        sol = solve_ivp(
            fun=_SEIRD, 
            t_span=(0, n_days), 
            y0=self.y0, 
            args=(self.beta, self.alpha, self.gamma, self.mu),
            method='RK45', 
            t_eval=np.arange(0, n_days, 1), 
            vectorized=True,
        )
        if self.sensitivity:
            I_ci = self.calculate_ci(self.sensitivity, sol.y[2])
            R_ci = self.calculate_ci(self.sensitivity, sol.y[3])
            D_ci = self.calculate_ci(self.sensitivity, sol.y[4])
            return sol.y[0], sol.y[1], I_ci, R_ci, D_ci
        return (sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4])
    
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