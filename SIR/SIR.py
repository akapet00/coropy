import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from SIR.utils import RMSE

def loss(params, confirmed_cases, recovered_cases, y0):
    def SIR(t, y):
        S, I, R = y
        N = S + I + R
        return [-beta*S*I/N, beta*S*I/N-gamma*I, gamma*I]
    beta, gamma = params
    size = len(confirmed_cases)
    sol = solve_ivp(SIR, (0, size), y0, method='RK45', t_eval=np.arange(0, size, 1), vectorized=True)
    return RMSE(sol.y[1], confirmed_cases) + RMSE(sol.y[2], recovered_cases)

class SIRModel(object):
    def __init__(self):
        pass

    def fit(self, confirmed_cases, recovered_cases, initial_conditions):
        self.y0 = initial_conditions
        opt = minimize(loss, [0.001, 0.001],
                    args=(confirmed_cases, recovered_cases, self.y0),
                    method='L-BFGS-B',
                    bounds=[(1e-7, 1.0), (1e-6, 1.0)])
        self.beta, self.gamma = opt.x
        print(f'R0 = {np.round(self.beta/self.gamma, 3)}')
        return self.beta, self.gamma

    def predict(self, n_days):
        def SIR(t, y):
            S, I, R = y
            N = S + I + R
            return [-self.beta*S*I/N, self.beta*S*I/N-self.gamma*I, self.gamma*I]
        sol = solve_ivp(SIR, (0, n_days), self.y0, method='RK45', t_eval=np.arange(0, n_days, 1), vectorized=True)
        return sol