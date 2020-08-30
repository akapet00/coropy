import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from covid_19.utils import mse, rmse, msle, mae


__all__ = ['SEIRModel', 'SEIRDModel']


def _SEIR(t, y, beta, delta, alpha, gamma):
    """Return the SEIR compartmental system values. For details check: 
    https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model
    
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

def _SEIRD(t, y, beta, delta, alpha, gamma, mu):
    """Return the SEIRD compartmental system values. For details check: 
    https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIRD_model
    
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
        -beta*S*I/N - delta*S*E, 
        beta*S*I/N - alpha*E + delta*S*E, 
        alpha*E - gamma*I - mu*I, 
        gamma*I,
        mu*I
    ]
    

class SEIRModel(object):
    """SEIR model class."""
    def __init__(self):
        """Constructor."""
        pass

    def fit(self, active_cases, removed_cases, initial_conditions):
        """Fit SEIR model.
        
        Parameters
        ----------
        active_cases: numpy.ndarray
            Time series of currently active infected individuals.
        removed_cases: numpy.ndarray
            Time series of recovered+deceased individuals.
        initial_conditions: list
            Values of S, E, I and R at the first day.
        
        Returns
        -------
        tuple
            Fitted epidemiological parameters: beta, delta, alpha and gamma rate.
        list
            Loss values during the optimization procedure.
        """
        loss = []
        def print_loss(p):
            """Optimizer callback."""
            loss.append(
                SEIRModel._loss(p, active_cases, removed_cases, initial_conditions)
            )
            
        self.y0 = initial_conditions
        opt = minimize(
            fun=SEIRModel._loss, 
            x0=[0.001, 0.001, 0.001, 0.001],
            args=(active_cases, removed_cases, self.y0),
            method='L-BFGS-B',
            bounds=[(1e-5, 1.0), (1e-5, 1.0), (1e-5, 1.0), (1e-5, 1.0),],
            options={'maxiter': 1000, 'disp': True},
            callback=print_loss,
        )
        self.beta, self.delta, self.alpha, self.gamma = opt.x
        return (self.beta, self.delta, self.alpha, self.gamma), loss

    def predict(self, n_days):
        """Forecast S, E, I and R based on the fitted epidemiological parameters.
        
        Parameters
        ----------
        n_days : int
            Number of days in future.
            
        Returns
        -------
        tuple
            S, E, I and R values forecast for n_days in future.
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
        return (sol.y[0], sol.y[1], sol.y[2], sol.y[3])
    
    @staticmethod
    def _loss(params, active_cases, removed_cases, initial_conditions, loss_fn='mse'):
        """Calculate and return the loss function between actual and predicted values.
        
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
            Loss function is `mse` by default, choose between `mse`, `rmse` and `msle`.
                
        Returns
        -------
        float
            Loss between the actual and predicted value of confirmed and recovered individuals.
        """
        if loss_fn == 'mse':
            loss_fn = mse
        elif loss_fn == 'rmse':
            loss_fn = rmse
        elif loss_fn == 'msle':
            loss_fn = msle
        elif loss_fn == 'mae':
            loss_fn = mae
        else:
            raise ValueError('Given loss function is not supported.')
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
        return loss_fn(sol.y[2], active_cases) + loss_fn(sol.y[3], removed_cases)


class SEIRDModel(object):
    """SEIR model class."""
    def __init__(self):
        """Constructor."""
        pass

    def fit(
        self, active_cases, recovered_cases, death_cases, initial_conditions, loss_fn='mse'):
        """Fit SEIRD model.
        
        Parameters
        ----------
        active_cases: numpy.ndarray
            Time series of currently active infected individuals.
        recovered_cases: numpy.ndarray
            Time series of recovered individuals.
        death_cases: numpy.ndarray
            Time series of deceased individuals.
        initial_conditions: list
            Values of S, E, I, R and D at the first day.
        loss_fn : str, optional
            Loss function is `mse` by default.
        
        Returns
        -------
        tuple
            Fitted epidemiological parameters: beta, delta, alpha, gamma and mu rate.
        list
            Loss values during the optimization procedure.
        """
        loss = []
        def print_loss(p):
            """Optimizer callback."""
            loss.append(
                SEIRDModel._loss(p, active_cases, recovered_cases, death_cases, initial_conditions)
            )
            
        self.y0 = initial_conditions
        opt = minimize(
            fun=SEIRDModel._loss, 
            x0=[0.1, 0.1, 0.1, 0.01, 0.01],
            args=(active_cases, recovered_cases, death_cases, self.y0, loss_fn),
            method='L-BFGS-B',
            bounds=[(1.e-5, 10.), (1.e-5, 10.), (1.e-5, 10.), (1.e-5, 10.), (1.e-5, 10.),],
            options={'disp': True, 'maxiter': 1000},
            callback=print_loss,
        )
        self.beta, self.delta, self.alpha, self.gamma, self.mu = opt.x
        return (self.beta, self.delta, self.alpha, self.gamma, self.mu), loss

    def predict(self, n_days):
        """Forecast S, E, I and R based on the fitted epidemiological parameters.
        
        Parameters
        ----------
        n_days : int
            Number of days in future.
            
        Returns
        -------
        tuple
            S, E, I and R values forecast for n_days in future.
        """
        sol = solve_ivp(
            fun=_SEIRD, 
            t_span=(0, n_days), 
            y0=self.y0, 
            args=(self.beta, self.delta, self.alpha, self.gamma, self.mu),
            method='RK45', 
            t_eval=np.arange(0, n_days, 1), 
            vectorized=True,
        )
        return (sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4])
    
    @staticmethod
    def _loss(params, active_cases, recovered_cases, death_cases, initial_conditions, loss_fn='mse'):
        """Calculate and return the loss function between actual and predicted values.
        
        Parameters
        ----------
        params : list
            Values of beta, delta, alpha and gamma rates.
        active_cases : numpy.ndarray
            Time series of currently active infected individuals.
        recovered_cases : numpy.ndarray
            Time series of recovered individuals.
        death_cases : numpy.ndarray
            Time series of deceased individuals.
        initial_conditions : list
            Values of S, E, I, R and D at the first day.
        loss_fn : str, optional
            Loss function is `mse` by default, choose between `mse`, `rmse` and `msle`.
        
        Returns
        -------
        float
            Loss between the actual and predicted value of confirmed, recovered and deceased individuals.
        """
        if loss_fn == 'mse':
            loss_fn = mse
        elif loss_fn == 'rmse':
            loss_fn = rmse
        elif loss_fn == 'msle':
            loss_fn = msle
        elif loss_fn == 'mae':
            loss_fn = mae
        else:
            raise ValueError('Given loss function is not supported.')
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
        return loss_fn(sol.y[2], active_cases) + loss_fn(sol.y[3], recovered_cases) + loss_fn(sol.y[4], death_cases)