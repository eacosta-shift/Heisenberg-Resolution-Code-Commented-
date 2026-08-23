#%% 
import numpy as np
from numpy import abs
import matplotlib.pyplot as plt

#* Variables
phi=np.linspace(0,2*np.pi,500)              #? φ
G=1                                         #? Gain
T=np.array([1,0.8,0.6])[:,None]             #? |τ|^2
t_s1=np.sqrt(T)                             #? τ_s1
t_i1=np.sqrt(1)                             #? τ_i1
t_s2=np.sqrt(1)                             #? τ_s2

U=np.cosh(G)
V=np.sinh(G)

#* Average number of output signal photons: <N_{s2}^{α=0}>
N0 = abs(t_s2)**2*abs(V)**2*  \
    (1-abs(t_i1)**2+abs(U)**2*(abs(t_s1)**2+abs(t_i1)**2+
     2*abs(t_s1)*abs(t_i1)*np.cos(phi)))

#* Quantum Fisher Information: F_Q
Fq=(4*abs(U)**2 * abs(V)**2)

#* Classical Fisher Information: F_C
dN0 = -2* abs(t_s1) * abs(t_i1) * abs(t_s2)**2 * abs(V)**2 * abs(U)**2 * np.sin(phi)        #? d/dφ ( <N_{s2}^{α=0}> ) 
Fc=1/(N0*(1+N0))*dN0**2

#* Inverse of Propagation of Errors: 1/<(Δφ)^2>
num = 4*abs(t_s2)**4*abs(t_s1)**2*abs(t_i1)**2 * abs(U)**4*abs(V)**4 * np.sin(phi)**2
dnum = N0*(1+N0)
InvError = num/dnum

#* Plots
colors = ["darkblue","orange","yellowgreen"]

fig,ax=plt.subplots()
ax.plot(phi,Fq*np.ones_like(phi),label=r"$F_Q$", color = "grey",linestyle="--",linewidth=2)
[ax.plot(phi,Fc[j,:],label=rf"$|\tau_{{s1}}|^2={T[j,0]}$", color = colors[j], linewidth=2) for j in range(T.size)]
ax.legend(loc="center left")

ax.set_xlim((0,2*np.pi))
ax.set_xticks(np.linspace(0,2*np.pi, 5))
ax.set_xticklabels([0, r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
ax.set_ylabel(r'Fisher Information ($F$)')
ax.set_xlabel(r'Phase ($\varphi$) [rad]')

ax.set_ylim((0,14))
# %%



 