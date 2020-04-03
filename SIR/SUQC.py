import numpy as np 
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import minimize
from SIR.utils import RMSE 

def loss(params, confirmed_cases, y0):
    def SUQC(t, y):
        S, U, Q, C = y
        #N = S + U + Q + C
        return [-alpha*U*S,             # S_t
                alpha*U*S - gamma_1*U,  # U_t
                gamma_1*U - beta*Q,     # Q_t
                beta*Q]                 # C_t
    alpha, gamma_1, beta = params
    size = len(confirmed_cases)
    sol = solve_ivp(SUQC, (0, size), y0, method='RK45', t_eval=np.arange(0, size, 1), vectorized=True)
    return RMSE(sol.y[3], confirmed_cases)

class SUQCModel(object):
    def __init__(self):
        pass

    def fit(self, confirmed_cases, initial_conditions):
        self.y0 = initial_conditions
        opt = minimize(loss, [0.1, 0.1, 0.1],
                    args=(confirmed_cases, self.y0),
                    method='BFGS', )
                    #bounds=[(1e-6, 3.0), (1e-6, 3.0), (1e-6, 3.0)]) # for L-BFGS-B
        self.alpha, self.gamma_1, self.beta = opt.x 
        print(opt.x)
        return self.alpha, self.gamma_1, self.beta

    def predict(self, n_days):
        def SUQC(t, y):
            S, U, Q, C = y
            #N = S + U + Q + C
            return [-self.alpha*U*S, 
                    self.alpha*U*S - self.gamma_1*U, 
                    self.gamma_1*U - self.beta*Q,
                    self.beta*Q]
        sol = solve_ivp(SUQC, (0, n_days), self.y0, method='RK45', t_eval=np.arange(0, n_days, 1), vectorized=True)
        return sol
