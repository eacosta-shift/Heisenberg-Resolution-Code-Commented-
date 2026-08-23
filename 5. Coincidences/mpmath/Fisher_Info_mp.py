#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# Use __file__ if running as a script, or fallback to the current working directory if in Jupyter
current_dir = Path(__file__).resolve() if '__file__' in locals() else Path.cwd()
# Go up two levels to get the root directory
root_dir = current_dir.parents[2] if '__file__' in locals() else current_dir.parent
# Safely inject the directory to Python's search path
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from coincidences import p_mn
from tqdm import tqdm

# ·································································································································································································································
# ·································································································································································································································
# ·································································································································································································································
#? Fisher Info (Eq. 3.6 of thesis manuscript) when using the coincidences probability dist. p_mn. The computattion 
#? of the Fisher Info. involves a derivative with respect to φ, which is taken numerically.
#? d/dφ(p_mn) = (p_mn(φ+dφ) - p_mn(φ-dφ)) /(2φ)

nPoints=200
pMax=15
G=0.5                               #? Gain
T=1                                 #? |τ|^2
t_s1=np.sqrt(T)                     #? τ_s1
t_i1=np.sqrt(T)                     #? τ_i1
t_s2=np.sqrt(T)                     #? τ_s2
t_i2=np.sqrt(T)                     #? τ_i2  

dphi=0.001
phi_vec=np.linspace(0,2*np.pi-dphi,nPoints) 
pm1=np.zeros((nPoints,pMax+1,pMax+1))   #? Probability at φ - dφ
p0=np.zeros((nPoints,pMax+1,pMax+1))    #? Probability at φ
p1=np.zeros((nPoints,pMax+1,pMax+1))    #? Probability at φ + dφ

# Fisher Info. in the case of Singles detection
f=np.zeros(nPoints)

for i in tqdm(range(nPoints)):

    phi=np.array((phi_vec[i],phi_vec[i]+dphi,phi_vec[i]-dphi))
    
    U=np.cosh(G)
    V=np.sinh(G)

   #* Average number of output signal photons (no seed): <N_{s2}^{α=0}>
    Ns2 = abs(t_s2)**2*abs(V)**2*    \
        (1+abs(t_s1)**2+(abs(t_s1)**2+abs(t_i1)**2)*abs(V)**2+
        2*abs(t_s1)*abs(t_i1)*abs(U)**2*np.cos(phi)) 

    #* Average number of output idler photons (no seed): <N_{i2}^{α=0}>
    Ni2 = abs(t_i2)**2*abs(V)**2*    \
        (1+abs(t_i1)**2+(abs(t_s1)**2+abs(t_i1)**2)*abs(V)**2+
        2*abs(t_s1)*abs(t_i1)*abs(U)**2*np.cos(phi))  

    #* |γ|^2 (Eq. 5.2 in thesis manuscript)
    g2 = 4 * t_i2**2 * t_s2**2 * abs(U)**2 * abs(V)**2 *(
        (t_i1**2-1)**2 - (t_i1**2-1)*(t_s1**2+t_i1*t_s1*np.cos(phi)) * 2*abs(V)**2
        + (t_i1**4 + t_i1**2*t_s1**2 + 2*t_i1**3*t_s1*np.cos(phi)) * abs(U)**4
        + (t_s1**4 + t_i1**2*t_s1**2 + 2*t_s1**3*t_i1*np.cos(phi)) * abs(V)**4
        + (  abs(V)**2*np.cos(phi)*(t_s1*t_i1*(t_s1**2+t_i1**2)+2*t_s1**2*t_i1**2*np.cos(phi))
            -(t_i1**2-1)*(t_i1**2+t_i1*t_s1*np.cos(phi))         ) * 2*abs(U)**2     )

    a = Ns2 + 1
    b = Ni2 + 1
    x = g2/(4*a*b)
    p0[i,:,:]=p_mn(pMax,a[0],b[0],x[0])
    p1[i,:,:]=p_mn(pMax,a[1],b[1],x[1])
    pm1[i,:,:]=p_mn(pMax,a[2],b[2],x[2])

    #? #* d/dφ ( <N_{s2}^{α=0}> ) 
    dNs2 = -2* abs(t_s1) * abs(t_i1) * abs(t_s2)**2 * abs(V)**2 * (1+abs(V)**2)*np.sin(phi[0])
    #? Fisher Info. in the case of Singles detection (for comparisson)
    f[i]=1/(Ns2[0]*(1+Ns2[0]))*dNs2**2


p0_inv=np.zeros_like(p0)
p0_inv[p0!=0]=1/p0[p0!=0]
F = np.sum(p0_inv*((p1-pm1)/(2*dphi))**2,axis=(1,2))    #? d/dφ(p_mn) = (p_mn(φ+dφ) - p_mn(φ-dφ)) /(2φ)
fig,ax=plt.subplots()
ax.plot(phi_vec,(4*abs(U)**2*abs(V)**2)*np.ones_like(phi_vec),color="grey",linestyle="--",label=r"$F_Q$", linewidth= 2)
ax.plot(phi_vec,f,color="deepskyblue",label=r"Singles", linewidth=2)
ax.plot(phi_vec,F,color="darkblue",label=r"Coincidences",linewidth=3, linestyle=":")
ax.set_xticks(np.linspace(0,2*np.pi, 5))
ax.set_xticklabels([0, r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
ax.set_xlim((0,2*np.pi))
ax.set_xlabel(r"Phase ($\varphi$) [rad]")
ax.set_ylabel(r"Fisher Information ($F$)")
ax.legend(handlelength=1, loc="lower right")
# %%
