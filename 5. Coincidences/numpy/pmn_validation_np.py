#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyadd,polymul,polymulx,polyder,polyfromroots
from scipy.special import factorial

def horner_eval(coeffs, x):
    "Efficient Poly Evaluation"
    result = 0
    for c in reversed(coeffs):
        result = result * x + c
    return result

def r_all(oMax,x):
    "Compute all r^m_j(x) up to m=j=oMax."
    "Use r[m,j] to call an element of r^m_j(x)]"
    Cj=np.zeros((oMax+1,oMax+1,1))
    [j,m,_]=np.indices((oMax+1,oMax+1,1))
    mask=m>=j
    Cj[mask]=(-1)**j[mask]/factorial(j[mask])**2*factorial(m[mask])/factorial((m-j)[mask])*(x)**j[mask]
    polys=np.zeros((oMax+1,1,oMax+1))
    for i in range(oMax+1):
        roots=-np.r_[1:i+1]
        polys[i,0,0:i+1]=polyfromroots(roots)
    r=Cj*polys
    return np.sum(r, axis=0)

def f_all(oMax,x):
    "Computes all the function f_n(x) up to n = oMax. Uses Horner Eval rather than polyval for improved numerical precision"
    f=np.zeros([oMax+1])
    f[0]=1/(1-x)
    Q=[np.array([1])]
    for n in range(1,oMax+1):
        Q.append(
            polymulx(polyadd(polymul(np.array([1,-1]),polyder(Q[n-1])),n*Q[n-1]))
        )
        f[n]=horner_eval(Q[n],x)/(1-x)**(n+1)
    return f

def p_mn(nMax,a,b,x):
    "Computes all probabilities p_mn up to m=n=nMax"
    ra=r_all(2*nMax,1/a)
    rb=r_all(2*nMax,1/b)
    f=f_all(2*nMax,x)

    q=np.zeros((nMax+1,nMax+1,2*nMax+1))
    [m,n,l]=np.indices(q.shape)  #? See eq. 5.6 of thesis manuscript to understand indices 
    mask=l<=(m+n)               #? r^u_d(x)=0 whenever d>u
    q[mask]=f[l[mask]]          #? mask avoids assigns

    s=np.zeros((nMax+1,nMax+1,2*nMax+1,2*nMax+1))
    [m,n,l,j]=np.indices(s.shape) #? See eq. 5.6 of thesis manuscript to understand indices 
    mask=j<=l
    s[mask]=ra[m[mask],j[mask]]*rb[n[mask],(l-j)[mask]]
    s=np.sum(s,axis=3) #? sum over j

    return 1/(a*b) * np.sum(q*s,axis=2) #? p = sum over l

# ·································································································································································································································
# ·································································································································································································································
# ·································································································································································································································

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
