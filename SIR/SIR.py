import numpy as np 

class SIR(object):
    """SIR disease model

    S' = - \beta * S * I
    I' = \beta * S * I - \nu * I 
    R' = \nu * I
    """
    def __init__(self, beta, nu, S0, I0, R0):
        if isinstance(nu, (float, int)):
            self.nu = lambda t: nu 
        elif callable(nu, (float, int)):
            self.nu = nu 
        
        if isinstance(nu, (float, int)):
            self.beta = lambda t: nu 
        elif callable(beta, (float, int)):
            self.beta = beta 

        self.initial_conds = [S0, I0, R0]

    def __call__(self, u, t):
        S, I, _ = u 
        return np.asarray([
            -self.beta(t) * S * I,
            self.beta(t) * S * I - self.nu(t) * I,
            self.nu(t) * I
        ])

class solver:
    """ODEsolver superclass

    Any classes inheriting from this superclass must implement
    advanc() method.
    """
    def __init__(self, f):
        self.f = f 

    def advance(self):
        raise NotImplementedError

    def set_initial_conds(self, U0):
        if isinstance(U0, (int, float)):
            # Scalaer ODE
            self.number_of_equations = 1
            U0 = float(U0)
        else:
            # System of ODEs
            U0 = np.asarray(U0)
            self.number_of_equations = U0.size 
        self.U0 = U0

    def solve(self, time_points):
        self.t = np.asarray(time_points)
        n = self.t.size 
        self.u = np.empty((n, self.number_of_equations))
        self.u[0, :] = self.U0

        # integrate 
        for i in range(n-1):
            self.i = i 
            self.u[i+1] = self.advance()

        return self.u[:i+2], self.t[:i+2]

class ForwardEuler(solver):
    def advance(self):
        u, f, i, t = self.u, self.f, self.i, self.t 
        dt = t[i+1] - t[i]
        return u[i, :] + dt * f(u[i, :], t[i]) 
