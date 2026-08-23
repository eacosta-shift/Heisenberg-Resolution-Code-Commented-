#%% 
import numpy as np
from numpy import abs
import matplotlib.pyplot as plt

#* Variables
phi=np.linspace(0,2*np.pi,100)              #? φ
G=1                                         #? Gain
T=0.8                                       #? |τ|^2
t_s1=np.sqrt(T)                             #? τ_s1
t_i1=np.sqrt(T)                             #? τ_i1
t_s2=np.sqrt(T)                             #? τ_s2
a_i=1                                       #? α_i

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

#* Quantum Fisher Information: F_Q
Fq=(4*abs(U)**2 * abs(V)**2 * (1+abs(a_i)**2))

#* Inverse of Propagation of Errors: 1/<(Δφ)^2>
num = 4*abs(t_s2)**4*abs(t_s1)**2*abs(t_i1)**2 * abs(U)**4*abs(V)**4 * (1+abs(a_i)**2)**2 * np.sin(phi)**2
dnum = N0*(1+N0) + (Na-N0)*(1+2*N0)
InvError = num / dnum

#* Plots

fig,ax=plt.subplots()
ax.plot(phi,Fq*np.ones_like(phi),label=r"$F_Q$", color = "grey",linestyle="--")
ax.plot(phi,InvError,":",label=r"(Prop. Errors)$^{-1}$", color = "orangered",linewidth=2)
ax.legend(loc="center left")

ax.set_xlim((0,2*np.pi))
ax.set_xticks(np.linspace(0,2*np.pi, 5))
ax.set_xticklabels([0, r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])

ax.set_xlabel(r'Phase ($\varphi$) [rad]')
ax.set_ylabel(r'Fisher Information ($F$)')


# %%
