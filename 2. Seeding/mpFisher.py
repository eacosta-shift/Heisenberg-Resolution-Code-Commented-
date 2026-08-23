#%%

import numpy as np
from numpy import abs
import matplotlib.pyplot as plt
from mpmath import mp, mpf, exp, laguerre
from tqdm import tqdm

#* Variables
phi=np.linspace(0,2*np.pi,500)              #? φ
G=1                                         #? Gain
T=0.6                                       #? |τ|^2
t_s1=np.sqrt(T)                             #? τ_s1
t_i1=np.sqrt(T)                             #? τ_i1
t_s2=np.sqrt(T)                             #? τ_s2
a_i=10                                      #? α_i

U=np.cosh(G)
V=np.sinh(G)

#* Average number of output signal photons (no seed): <N_{s2}^{α=0}>
N0 = abs(t_s2)**2*abs(V)**2*  \
    (1-abs(t_i1)**2+abs(U)**2*(abs(t_s1)**2+abs(t_i1)**2+
    2*abs(t_s1)*abs(t_i1)*np.cos(phi)))    

#* Average number of output signal photons (seed): <N_{s2}^{α}>
Na= abs(t_s2)**2*abs(V)**2*  \
    (1-abs(t_i1)**2+(1+abs(a_i)**2)*abs(U)**2*(abs(t_s1)**2+abs(t_i1)**2+
    2*abs(t_s1)*abs(t_i1)*np.cos(phi)))

#* d/dφ ( <N_{s2}^{α=0}> )    
dN0 = -2* abs(t_s1) * abs(t_i1) * abs(t_s2)**2 * abs(V)**2 * abs(U)**2 * np.sin(phi)

#* Inverse of Propagation of Errors: 1/<(Δφ)^2>
num = 4*abs(t_s2)**4*abs(t_s1)**2*abs(t_i1)**2 * abs(U)**4*abs(V)**4 * (1+abs(a_i)**2)**2 * np.sin(phi)**2
dnum = N0*(1+N0) + (Na-N0)*(1+2*N0)
InvError = num / dnum


def FisherInfo(mMax, a_i, N0, Na, dN0, phi, dps=20):
    '''
    Computes the Fisher Information for the case of seeding using the library mpmath
    (Python library for real and complex floating-point arithmetic with arbitrary precision).
    
    However, numpy and mpmath are not compatible. mpmath doesn't allow to vectorize all operations, 
    and it is not possible to convert numpy objects such as ndarray or np.float64 into a
    mpf (“multiple-precision floating-point”, mpmath object).

    To work around this, any np.float64 must first be converted to a python float before finally
    being converted into a mpf object. Furthermore, ndarrays with dtype=object are needed to store
    the mpf's.

    The dps argument controls how many decimal digits of accuracy mpmath uses in its calculations.

    WORKS ONLY FOR A SINGLE VALUE OF α_i, NOT A VECTOR.
    '''

    mp.dps=dps

    q = phi!=np.pi      # 'query'

    # Argument of Laguerre Polynomials, λ in manuscript (eq. 4.21)
    X = np.zeros_like(N0)
    X[~q] = (np.ones_like(np.asarray(phi)) * a_i**2)[~q]
    X[q] = (Na[q]-N0[q])/(N0[q]*(N0[q]+1))
    X = X.astype(float)

    # Term that contains 0/0 indeterminacy at φ=π. Needs to be conditionally determined to avoid numerical 0/0  
    Aux = np.zeros_like(N0)
    Aux[~q] = 0
    Aux[q] = dN0[q]/N0[q]**2
    Aux = Aux.astype(float)

    a_i = mpf(a_i)
    N0  = N0.astype(float)
    Na  = Na.astype(float)
    P  = np.zeros((mMax+1,N0.size), dtype=object)
    dP = np.zeros((mMax+1,N0.size), dtype=object)

    for i in tqdm(range(mMax+1)):
        for j in range(N0.size):
            m   = mpf(i)
            x   = mpf(X[j])
            n0  = mpf(N0[j])
            na  = mpf(Na[j])
            aux = mpf(Aux[j])

            P[i,j]  = 1/(1+n0)* (n0 / (1 + n0))**m * exp(-n0*x) * laguerre(m,0,-x)

            dP[i,j] = P[i,j] * aux * (
                    n0/(1+n0)*(abs(a_i)**2-x*(2*n0+1))*(laguerre(m-1,1,-x)/laguerre(m,0,-x)+1)
                    -n0*(2+abs(a_i)**2-(1+m)/(1+n0)) + na  ) 

    sumandos = np.zeros_like(P, dtype=object)
    sumandos = (1/P) * dP**2
    return np.sum(sumandos,axis=0), sumandos

mMax = 1500
F, sumandos = FisherInfo(mMax, a_i, N0, Na, dN0, phi, dps=50)

fig, axs = plt.subplots(2, 1, layout="constrained", figsize=(6,9))
axs[0].plot(np.arange(mMax+1),sumandos[:,1],marker="v",markersize=1,linestyle="")
axs[0].set_xlim([0, mMax+1])
axs[0].set_xlabel("m")
axs[0].set_title(r"$\frac{1}{p_m}\left(\frac{\partial p_m}{\partial\varphi}\right)^2$")

axs[1].plot(phi, InvError, color = "orangered", label=r"(Prop. Errors)$^{-1}$")
axs[1].plot(phi, F, ".", color = "darkblue", label=r"Numerical $F_c$", markersize=3)
axs[1].legend()
axs[1].set_xlim([0, 2*np.pi])
axs[1].set_xticks(np.linspace(0, 2*np.pi, 5))
axs[1].set_xticklabels([0, r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
axs[1].set_xlabel(r'$\varphi$')
axs[1].set_ylabel(r'F')
axs[1].set_title(r"Numerical $F_c$")

# %%
