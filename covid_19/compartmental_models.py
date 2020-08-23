import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from covid_19.utils import mse


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
        Values of the SEIR compartmental model.
    """
    S, E, I, R = y
    N = S + E + I + R
    return [
        -beta*S*I/N - delta*S*E, 
        beta*S*I/N - alpha*E + delta*S*E, 
        alpha*E - gamma*I, 
        gamma*I,
    ]

    
def _loss(params, active_cases, removed_cases, initial_conditions):
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
    
    Returns
    -------
    float
        Loss between the actual and predicted value of confirmed and recovered individuals.
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
    return mse(sol.y[2], active_cases) + mse(sol.y[3], removed_cases) 


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
                _loss(p, active_cases, removed_cases, initial_conditions)
            )
            
        self.y0 = initial_conditions
        opt = minimize(
            fun=_loss, 
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