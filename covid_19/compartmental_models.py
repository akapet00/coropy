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

    def __init__(
        self, 
        loss_fn='mse',
        sensitivity=None,
        specificity=None,
        new_positives=None,
        total_tests=None,
        ):
        """Constructor.
        
        Parameters
        ----------
        loss_fn: str, optional
            Loss function is `mse` by default
        sensitivity : float, optional
            Measure of the proportion of positives that are correctly
            identified (e.g., the percentage of sick people who are
            correctly identified as being infected). Only if all the
            optional variables (`sensitivity`, `specifcity`,
            `new_positives` and `total_tests`) are defined, the .95
            confidence intervals is going to be caluclated.  
        specificity : float, optional
            Measure of the proportion of negatives that are correctly
            identified (e.g., the percentage of healthy people who are
            correctly identified as not infected).
        new_positives : numpy.ndarray, optional
            Time series of new daily infected individuals.
        total_tests : numpy.ndarray, optional
            Time series of new daily completed tests.
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
        if (sensitivity and specificity and 
                new_positives is not None and total_tests is not None):
            assert isinstance(sensitivity, (float, )), \
                'Sensitivity must be a floating point number in <0.5, 1]'
            assert isinstance(specificity, (float, )), \
                'Specificity must be a floating point number in <0.5, 1]'
            self.sensitivity = sensitivity
            self.specificity = specificity
            self.new_positives = new_positives
            self.total_tests = total_tests
        else:
            self.sensitivity = None
            self.specificity = None
            self.new_positives = None
            self.total_tests = None
    
    @staticmethod
    def calculate_ci(
        sensitivity, specificity, positives, actives, recoveries, total_tests
        ):
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
        specificity : float or numpy.ndarray
            Test specificity. If the sensitivity is different in
            different intervals, it should be stored in the array-like
            format where the length is the same as the length of the
            data.
        positives : numpy.ndarray
            Daily number of new confirmed positive infections.
        actives : numpy.ndarray
            Time series of currently active infected individuals.
        recoveries : numpy.ndarray
            Daily number of new confirmed recoveries.        
        total_tests : numpy.ndarray
            Daily number of tests.

        Returns
        -------
        tuple
            Tuple consisting two numpy array. The first array is the
            lower CI bound scaler, and the second row is the upper CI
            bound scaler.
        """
        std_sensitivity_err = np.sqrt(np.divide(
            (1 - sensitivity) * sensitivity, 
            positives, 
            out=np.zeros(positives.shape, dtype=float), 
            where=positives!=0,
        ))
        sensitivity_ci = 1.960 * std_sensitivity_err
        lower_bound_sensitivity = np.abs(sensitivity - sensitivity_ci)
        lower_bound_true_positives = lower_bound_sensitivity * positives ####
        cumulative_cases = np.cumsum(lower_bound_true_positives)
        lower_bound_active_infections = cumulative_cases - recoveries
        lower_bound_scaler = lower_bound_active_infections / actives
        
        negatives = total_tests - positives
        std_specificity_err = np.sqrt(np.divide(
            (1 - specificity) * specificity,
            negatives,
            out=np.zeros(negatives.shape, dtype=float), 
            where=negatives!=0,
        ))
        specificity_ci = 1.960 * std_specificity_err
        upper_bound_specificity = np.abs(specificity + specificity_ci)
        upper_bound_true_negatives = upper_bound_specificity * negatives ####
        upper_bound_false_negatives = negatives - upper_bound_true_negatives
        upper_bound_true_positives = upper_bound_false_negatives + positives
        cumulative_cases = np.cumsum(upper_bound_true_positives)
        upper_bound_active_infections = cumulative_cases - recoveries
        upper_bound_scaler = upper_bound_active_infections / actives
        return (lower_bound_scaler, upper_bound_scaler)
            

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
            t_span=(0, self.active_cases.size), 
            y0=self.y0, 
            args=(self.beta, self.delta, self.alpha, self.gamma),
            method='RK45', 
            t_eval=np.arange(0, self.active_cases.size, 1), 
            vectorized=True,
        )
        if (self.sensitivity and self.specificity and 
                self.total_tests is not None):
            (lower_bound_scaler, upper_bound_scaler) = self.calculate_ci(
                self.sensitivity, 
                self.specificity,
                self.new_positives,
                self.active_cases,
                self.removed_cases,
                self.total_tests,
                )
            I_ci = np.r_[
                (lower_bound_scaler*sol.y[2]).reshape(1, -1),
                sol.y[2].reshape(1, -1),
                (upper_bound_scaler*sol.y[2]).reshape(1, -1),
            ]
            return (sol.y[0], sol.y[1], I_ci, sol.y[3])
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

    def simulate(self):
        """Forecast S, E, I and R based on the fitted epidemiological
        parameters.
            
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
            (lower_bound_scaler, upper_bound_scaler) = self.calculate_ci(
                self.sensitivity, 
                self.specificity,
                self.new_positives,
                self.active_cases,
                self.recovered_cases,
                self.total_tests,
                )
            I_ci = np.r_[
                (lower_bound_scaler*sol.y[2]).reshape(1, -1),
                sol.y[2].reshape(1, -1),
                (upper_bound_scaler*sol.y[2]).reshape(1, -1),
            ]
            return (sol.y[0], sol.y[1], I_ci, sol.y[3], sol.y[4])
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