import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, fmin_l_bfgs_b
from covid_19.utils import RMSE

def lossSIR(params, confirmed_cases, recovered_cases, y0):
    def SIR(t, y):
        S, I, R = y
        N = S + I + R
        return [-beta*S*I/N,                        # dS/dt - change of susceptibles
                beta*S*I/N - gamma*I,               # dI/dt - change of infectious 
                gamma*I]                            # dR/dt - change of recovered/removed/dead/quarantineed
    beta, gamma = params
    size = len(confirmed_cases)
    sol = solve_ivp(SIR, (0, size), y0, method='RK45', t_eval=np.arange(0, size, 1), vectorized=True)
    return RMSE(sol.y[1], confirmed_cases) + RMSE(sol.y[2], recovered_cases) # loss func = l2||I(t), fitted infected|| + l2||R(t), fitted recovered||

class SIR(object):
    def __init__(self):
        pass

    def fit(self, confirmed_cases, recovered_cases, initial_conditions):
        self.y0 = initial_conditions
        opt = minimize(lossSIR, [0.001, 0.001],
                       args=(confirmed_cases, recovered_cases, self.y0),
                       method='L-BFGS-B',
                       bounds=[(1e-7, 1.0), (1e-6, 1.0)])
        self.beta, self.gamma = opt.x
        return self.beta, self.gamma

    def predict(self, n_days):
        def SIR(t, y):
            S, I, R = y
            N = S + I + R
            return [-self.beta*S*I/N, 
                    self.beta*S*I/N-self.gamma*I, 
                    self.gamma*I]
        sol = solve_ivp(SIR, (0, n_days), self.y0, method='RK45', t_eval=np.arange(0, n_days, 1), vectorized=True)
        return sol

######################################################################################################################################################

def lossSEIR(params, confirmed_cases, recovered_cases, y0):
    def SEIR(t, y):
        S, E, I, R = y
        N = S + E + I + R
        return [-beta*S*I/N - delta*S*E,            # dS/dt - change of susceptibles
                beta*S*I/N - alpha*E + delta*S*E,   # dE/dt - change of exposed (asymptomatic but still infectious)
                alpha*E - gamma*I,                  # dI/dt - change of infectious with symptoms of disease
                gamma*I]                            # dR/dt - change of removed from system 
    beta, delta, alpha, gamma = params
    size = len(confirmed_cases)
    sol = solve_ivp(SEIR, (0, size), y0, method='RK45', t_eval=np.arange(0, size, 1), vectorized=True)
    return RMSE(sol.y[2], confirmed_cases) + RMSE(sol.y[3], recovered_cases) # loss func = l2||I(t), fitted infected|| + l2||R(t), fitted recovered||


class SEIR(object):
    def __init__(self):
        pass

    def fit(self, confirmed_cases, recovered_cases, initial_conditions):
        self.y0 = initial_conditions
        opt = minimize(lossSEIR, [0.001, 0.001, 0.001, 0.001],
                       args=(confirmed_cases, recovered_cases, self.y0),
                       method='L-BFGS-B',
                       bounds=[(1e-4, 1.0), (1e-4, 1.0), (1e-4, 1.0), (1e-4, 1.0),],
                       options={'maxiter':15000,
                                'disp':False})
        self.beta, self.delta, self.alpha, self.gamma = opt.x
        return self.beta, self.delta, self.alpha, self.gamma

    def predict(self, n_days):
        def SEIR(t, y):
            S, E, I, R = y
            N = S + E + I + R
            return [-self.beta*S*I/N - self.delta*S*E,
                    self.beta*S*I/N - self.alpha*E + self.delta*S*E, 
                    self.alpha*E - self.gamma*I,                
                    self.gamma*I]  
        sol = solve_ivp(SEIR, (0, n_days), self.y0, method='RK45', t_eval=np.arange(0, n_days, 1), vectorized=True)
        return sol
    
class SEIQR(object):
    def __init__(self):
        pass