#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyadd,polymul,polymulx,polyder
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


nPoints=100
P = np.zeros((nPoints,3,3))
Q = np.zeros((nPoints,3,3))

G=0.5                                 #? Gain
t_i1=np.sqrt(0.8)                     #? τ_i1
t_s1=np.sqrt(0.8)                     #? τ_s2
t_i2=np.sqrt(0.8)                     #? τ_i2  
t_s2=np.sqrt(0.8)                     #? τ_s2
phi_vec=np.linspace(0,2*np.pi,nPoints)           

for i,phi in enumerate(phi_vec):

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
    p=p_mn(2,a,b,x)

    # ·································································································································································································································
    # ·································································································································································································································
    # ·································································································································································································································

    def horner_eval(coeffs, x):
        result = 0
        for c in reversed(coeffs):
            result = result * x + c
        return result
    
    def f_all(oMax,x):
        f=np.zeros([oMax+1])
        f[0]=1/(1-x)
        Q=[np.array([1])]
        for n in range(1,oMax+1):
            Q.append(
                polymulx(polyadd(polymul(np.array([1,-1]),polyder(Q[n-1])),n*Q[n-1]))
            )
            f[n]=horner_eval(Q[n],x)/(1-x)**(n+1)
        return f
    
    f=f_all(4,x)
    q=np.zeros((3,3))

    q[0,0] = 1/(a*b) * 1/(1-x)
    q[1,0] = 1/(a*b) *((1-1/a)*f[0]-1/a*f[1])
    q[0,1] = 1/(a*b) *((1-1/b)*f[0]-1/b*f[1])
    q[1,1] = 1/(a*b) * ((1-1/a)*(1-1/b)*f[0] - (1/a*(1-1/b)+1/b*(1-1/a))*f[1]+1/(a*b)*f[2])
    q[2,0] = 1/(a*b) * ((1-2/a+1/a**2)*f[0] + (3/(2*a**2)-2/a)*f[1] + 1/(2*a**2)*f[2])
    q[0,2] = 1/(a*b) * ((1-2/b+1/b**2)*f[0] + (3/(2*b**2)-2/b)*f[1] + 1/(2*b**2)*f[2])

    q[2,1] = 1/(a*b) * ( 
        (1-2/a+1/a**2)*(1-1/b)*f[0] 
        + ((3/(2*a**2)-2/a)*(1-1/b)-1/b*(1-2/a+1/a**2))*f[1]
        + (1/(2*a**2)*(1-1/b)-1/b*(3/(2*a**2)-2/a))*f[2]
        - 1/(2*a**2*b)*f[3]
    )

    q[1,2] = 1/(a*b) * ( 
        (1-2/b+1/b**2)*(1-1/a)*f[0] 
        + ((3/(2*b**2)-2/b)*(1-1/a)-1/a*(1-2/b+1/b**2))*f[1]
        + (1/(2*b**2)*(1-1/a)-1/a*(3/(2*b**2)-2/b))*f[2]
        - 1/(2*b**2*a)*f[3]
    )

    q[2,2] = 1/(a*b) * ( 
        (1-2/b+1/b**2)*(1-2/a+1/a**2)*f[0] 
        + ((1-2/a+1/a**2)*(3/(2*b**2)-2/b)+(1-2/b+1/b**2)*(3/(2*a**2)-2/a))*f[1]
        + (1/(2*a**2)*(1-2/b+1/b**2)+1/(2*b**2)*(1-2/a+1/a**2)+(3/(2*a**2)-2/a)*(3/(2*b**2)-2/b))*f[2]
        + (1/(2*b**2)*(3/(2*a**2)-2/a)+1/(2*a**2)*(3/(2*b**2)-2/b))*f[3]
        + 1/(4*a**2*b**2)*f[4]
    )

    P[i,:,:]=p
    Q[i,:,:]=q



fig,ax=plt.subplots(3,3,layout="constrained")
fig.set_size_inches([11.58,  6.57])

for i in range(3):
    for j in range(3):
        ax[i,j].plot(phi_vec,Q[:,i,j], label="Theoretical")
        ax[i,j].plot(phi_vec,P[:,i,j],linestyle="--",label="Algorithmic")
        ax[i,j].set_title(fr"$p_{{{i,j}}}$")
        ax[i,j].set_xticks(np.linspace(0,2*np.pi, 5))
        ax[i,j].set_xticklabels([0, r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
        ax[i,j].grid(True)
        ax[i,j].set_xlim((0,2*np.pi))
        ax[i,j].legend()
# %%
