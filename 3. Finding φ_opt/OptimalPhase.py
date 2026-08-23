import numpy as np
from numpy import abs
from mpmath import mp, mpf, sqrt, cosh, sinh

def NP(func):
    '''
    This function takes an mpmath function 'func' meant to operate on mpf objects, and outputs
    an user defined function that applies 'func' to every element of a numpy array.
    '''
    def wrapper(ndarray):

        if (isinstance(ndarray,np.ndarray)==True and isinstance(ndarray[0],mpf)==False):
            ndarray=np.asarray(ndarray,dtype=float)
        if (isinstance(ndarray,np.ndarray)==False and isinstance(ndarray,mpf)==False):
            ndarray=np.asarray(ndarray,dtype=float)

        out = np.empty_like(ndarray, dtype=object)
        for idx, x in np.ndenumerate(ndarray):
            out[idx] = func(mpf(x))
        if out.size==1:
            return out.item()
        else:
            return out
    return wrapper

Mpf   = NP(mpf)
Sqrt  = NP(sqrt)
Cosh  = NP(cosh)
Sinh  = NP(sinh)

def OptimalPhase(a_i=0,G=1,t_s1=1,t_i1=1,t_s2=1,dps=50):
    '''
    This function finds the optimal phase given a set of values for the parameter space, 
    as defined by eqs. 4.10 and 4.11 of the thesis manuscript.
    '''

    mp.dps = dps

    a_i  = Mpf(a_i)
    G    = Mpf(G)
    t_s1 = Mpf(t_s1)
    t_i1 = Mpf(t_i1)
    t_s2 = Mpf(t_s2)

    U = Cosh(G)
    V = Sinh(G)

    K2 = 4 * abs(t_s2*U*V)**4 * abs(t_s1*t_i1)**2 * (1+2*abs(a_i)**2)

    K1 = 1/2 * abs(t_s1*t_i1) * abs(t_s2)**2 * Sinh(2*G)**2 *        \
        (1 + abs(a_i)**2 + 2*abs(t_s2*V)**2*(1 + abs(a_i)**2 + abs(t_s1*U)**2*(1+2*abs(a_i)**2) + abs(t_i1)**2*(abs(V)**2+abs(a_i)**2*Cosh(2*G))))

    K0 = abs(t_s2*V)**2 * ( (1+abs(t_i1*V)**2+abs(t_s1*U)**2) * (1+abs(t_s2*V)**2*(1+abs(t_s1*U)**2+abs(t_i1*V)**2)) +
                        abs(a_i*U)**2 * (abs(t_s1)**2+abs(t_i1)**2) * (1+2*abs(t_s2*V)**2*(1+abs(t_s1*U)**2+abs(t_i1*V)**2)))  
    
    kappa = abs(Sqrt((K0-K1+K2)*(K0+K1+K2)))
    x = np.asarray( (kappa-K0-K2)/K1 )
    phi0 = np.arccos(x.astype(float))

    return phi0
