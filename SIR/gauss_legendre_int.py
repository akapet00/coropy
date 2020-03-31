import numpy as np

def legendre(x, degree) -> np.array:
    """ Return n-th degree Legendre polynomial. 

    arguments
    ---------
    x (numpy.array) - array of values in which the polynomial will be calculated
    degree (int) - degree of Legendre polynomial to be returned
    """
    if degree == 0:
        return 1
    elif degree == 1:
        return x 
    else:
        # Bonnete's recursion formula
        return (2*degree-1)/degree * x * legendre(x, degree-1) -\
               (degree-1)/degree * legendre(x, degree-2)

def legendre_x(x, degree) -> np.array:
    """ Return n-th degree Legendre polynomial. 

    arguments
    ---------
    x (numpy.array) - array of values in which the polynomial will be calculated
    degree (int) - degree of Legendre polynomial to be returned
    """
    if degree == 0:
        return 0
    elif degree == 1:
        return 1 
    else:
        # Derivative of Bonnete's recursion formula
        return (degree / (x**2-1)) * (x * legendre(x, degree) - legendre(x, degree-1))

def legendreRoots(degree) -> np.array:
    """ Return roots of Legendre Polynomials using Newton-Raphson method.

    arguments
    ---------
    degree (int) - degree of Legendre polynomial
    """
    if isinstance(degree, int) != True:
        raise ValueError('Degree must be a positive integer.')
    if degree == 1:
        return 0
    else:
        # Newton's method
        roots = []
        for i in range(1, int(degree/2)+1):
            x = np.cos(np.pi * (i - 0.25)/(degree + 0.5))
            for _ in range(100):
                x -= legendre(x, degree)/legendre_x(x, degree)
            roots.append(x)  
        roots = np.array(roots)
        if degree%2==0:
            roots = np.r_[-roots, np.flip(roots)]
        else: roots = np.r_[-roots, [0], np.flip(roots)]
    return roots.squeeze()


def gaussLegendreWeights(n_points) -> tuple:
    """ Return values of collocation points together with ssociated weights.

    arguments
    ---------
    n_points (int32) - degree of Gauss-Legendre quadrature
    """
    psis = legendreRoots(n_points)
    ws = 2 / ((1 - psis**2) * legendre_x(psis, n_points)**2)
    return (psis, ws)

def gaussQuadrature(func, a, b, n_points, printout=False) -> float:
    """ Return the value of the integral of given function.

    arguments
    ---------
    a (int32, float32) - left border of integration domain
    b (int32, float32) - right border of integration domain
    n_points (int32) - degree of Gauss-Legendre quadrature
    printout (Bool) - print the roots and weights
    """
    psis, ws = gaussLegendreWeights(n_points)
    if printout: print(f'Roots: {psis}\n Weights: {ws}')
    xs = (b-a)/2 * psis + (a+b)/2
    I = 0
    for w, x in zip(ws, xs):
        I += (b-a)/2 * w * func(x)
    return I, xs

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    
    funcstring = 'x**2 - 1'
    func = lambda x: eval(funcstring) 
    a = 0 
    b = 1 
    n_points = 10

    I, xs = gaussQuadrature(func, a, b, n_points, printout=True)

    print(f'Integral of function {funcstring} is {I} using {n_points}-th order Gaussian quadrature')

    x_axis = np.linspace(-2, 2, 100)
    x_domain = np.linspace(a, b)

    plt.plot(x_axis, func(x_axis), 'k--')
    plt.plot(x_domain, func(x_domain), 'r-')
    plt.vlines(1, -1, 0, color='red')
    plt.hlines(-1, 0, 1, color='red')
    plt.fill_between(x_domain, func(x_domain), -1, color='red', alpha=0.3, label='integration area')

    # function approximation
    xs = np.array(xs)
    plt.plot(xs, func(xs), marker='o', linestyle='None', label='function approximation')

    # integration
    plt.arrow(-0. , -0.1, 0.8, -0.7)
    plt.text(-0.5, 0.0, 'Integral = {}'.format(np.round(I, 3)),
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

    plt.title('Gauss-Legendre integration using {} points'.format(n_points))
    plt.axis([-2, 2, -1.2, 0.5])
    plt.grid()
    plt.legend(loc='best')
    plt.show()
