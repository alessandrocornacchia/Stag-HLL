import numpy as np
import matplotlib.pyplot as plt

# add new functions here, and define below the default. 
def const(a,t):
    return (a,None) if np.isscalar(t) else (a*np.ones(len(t)), a)

def two_level(a1,a2,duty,T,t):
    
    if np.isscalar(t):
        if t % T < duty * T:
            return (a1, a1)
        else:
            return (a2,a2)
    else:
        lambda_ = []
        for ti in t:
            if ti % T < duty * T:
                lambda_.append(a1)
            else:
                lambda_.append(a2)
        return (lambda_, None)

def linear_ramp(T,a,t):
    fmax = a*T
    return (a*t if t < T else a*T, fmax)

def sinusoid(f,a,t):
    fmax = 2*a
    return (a * np.sin(2*np.pi*f*t) + a, fmax)

def modulated_triangular(f,a,t):
    fmax = a
    return (a * t * np.cos(2*np.pi*f*t), fmax)

################ default ####################
# It is not mandatory to provide values for the time time parameter 't',
# as it will be subsituted by simulation time instances at runtime. 
# N.B. even if no value is provided, include the literal character 't'.
LAMBDA_T = 'const(1000,t)'
####################################

# plot test functions and the average arrival rate on one period
if __name__ == '__main__':
    T=10
    n = 1e3
    t = np.linspace(0,T,int(n))
    LAMBDA_T = 'sinusoid(0.1,500,t)'
    y,_ = eval(LAMBDA_T)
    plt.plot(t, y)
    print('Numerical average arrival rate', 1/T * np.trapz(y,t))
    plt.show()

    