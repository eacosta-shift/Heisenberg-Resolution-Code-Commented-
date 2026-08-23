#%%
#%matplotlib widget
import numpy as np
from numpy import abs
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre

#* Variables
phi=np.linspace(0,2*np.pi,400)              #? φ
G=1                                         #? Gain
T=0.8                                       #? |τ|^2
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


def FisherInfo(mMax, a_i, N0, Na, dN0, phi):

    q = phi!=np.pi      # 'query'

    # Argument of Laguerre Polynomials, λ in manuscript (eq. 4.21)
    X = np.zeros_like(N0)
    X[~q] = (np.ones_like(np.asarray(phi)) * a_i**2)[~q]
    X[q] = (Na[q]-N0[q])/(N0[q]*(N0[q]+1))

    # Term that contains 0/0 indeterminacy at φ=π. Needs to be conditionally determined to avoid numerical 0/0  
    Aux = np.zeros_like(N0)
    Aux[~q] = 0
    Aux[q] = dN0[q]/N0[q]**2

    m = np.arange(mMax+1)[:,None]

    P = 1/(1+N0)* (N0 / (1 + N0))**m * np.exp(-N0*X) * eval_genlaguerre(m,0,-X)

    dP = P * Aux * (
            N0/(1+N0)*(abs(a_i)**2-X*(2*N0+1))*(eval_genlaguerre(m-1,1,-X)/eval_genlaguerre(m,0,-X)+1)
            -N0*(2+abs(a_i)**2-(1+m)/(1+N0)) + Na  ) 
    
    sumandos = np.zeros_like(P)
    sumandos[P!=0] = (1/P[P!=0]) * dP[P!=0]**2          # Avoid division by 0 
    return np.sum(sumandos,axis=0), sumandos[:,1]       # Add all summands together for each combination of parameters being considered


mMax = 1500
F, sumandos = FisherInfo(mMax, a_i, N0, Na, dN0, phi)

fig, axs = plt.subplots(2, 1, layout="constrained", figsize=(6,9))
axs[0].plot(np.arange(mMax+1), sumandos, marker="v", markersize=1, linestyle="")
axs[0].set_xlim([0, mMax+1])
#axs[0].set_ylim(bottom = 0)
axs[0].set_xlabel("m")
axs[0].set_title(r"$\frac{1}{p_m}\left(\frac{\partial p_m}{\partial\varphi}\right)^2$")

axs[1].plot(phi, InvError, color = "tomato", label=r"(Error Propagation)$^{-1}$")
axs[1].plot(phi, F, color = "darkblue", label=r"Numerical $F_c$", markersize=3)
axs[1].legend()
axs[1].set_xlim([0, 2*np.pi])
axs[1].set_xticks(np.linspace(0, 2*np.pi, 5))
axs[1].set_xticklabels([0, r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
axs[1].set_xlabel(r'$\varphi$')
axs[1].set_ylabel(r'F')
axs[1].set_title(r"Numerical $F_c$")



'''
What is going on with the classical Fisher Info? Why are there inaccuracies as φ goes to π?


As discussed in other sections of the code and documentation, the number of summands 
required for the sum of the classical Fisher info to converge increases as φ goes to 0.

On the other hand, when the φ goes to π, the number of summands that are different from 
zero starts decreasing (goes to zero), and only a few terms are enough for the sum to converge. E.g as 
φ goes to π, the classical Fisher Info may be obtained by considering only the first 5 (or even 
less) terms.

However, the NUMBER OF TERMS necessary to obtain the Fisher Info AS A FUNCTION OF φ
is something that I have no idea whatsoever how to determine. Because of this, it is
necessary to set a number of summands mMax that will be computed for ALL values of φ.

In principle, by computing as many summands as required for the Fisher info to converge 
when φ tends to zero, this same number of summands mMax is enough for the sum to converge for
ALL values of φ, since as φ goes from 0 to π, the number of summands required for convergence decreases. 
This reasoning is mathematically safe.

However, when implementing this, mMax needs to be greater than 1000 so that F_c converges
at φ = 0, even in conservative situations. Hence, the same number of summands (>1000) is
computed also when φ=π, where only a few summands (~1) are different from 0.

What this means is that as φ goes to π, operations are yielding ever decreasing 
values that all tend to zero faster and faster. As m increases, this eventually leads to numerical
inaccuracies, such as overflow. This is why it becomes necessary to resort to 
an arbitrary precision arithmetic package (mpmath). 
'''
# %%
