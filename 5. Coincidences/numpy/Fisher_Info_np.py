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
    "Computes all the function f_n(x) up to n = oMax. Uses Horner Eval rather than polyval for improved numerical precision "

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

#? Fisher Info (Eq. 3.6 of thesis manuscript) when using the coincidences probability dist. p_mn. The computattion 
#? of the Fisher Info. involves a derivative with respect to φ, which is taken numerically.
#? d/dφ(p_mn) = (p_mn(φ+dφ) - p_mn(φ-dφ)) /(2φ)

nPoints=150
pMax=12
G=0.5                               #? Gain
t_s1=np.sqrt(1)                     #? τ_s1
t_i1=np.sqrt(1)                     #? τ_i1
t_s2=np.sqrt(1)                     #? τ_s2
t_i2=t_s2                           #? τ_i2  

dphi=0.001
phi_vec=np.linspace(0,2*np.pi-dphi,nPoints) 
p0=np.zeros((nPoints,pMax+1,pMax+1))   #? Probability at φ
p1=np.zeros((nPoints,pMax+1,pMax+1))   #? Probability at φ + dφ
pm1=np.zeros((nPoints,pMax+1,pMax+1))  #? Probability at φ - dφ

f=np.zeros(nPoints) #? Fisher Info. in the case of Singles detection (for comparisson)

for i in range(nPoints):

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
    pm1[i,:,:]=p_mn(pMax,a[2],b[2],x[2]) #? Probability at φ
    p0[i,:,:]=p_mn(pMax,a[0],b[0],x[0])  #? Probability at φ + dφ
    p1[i,:,:]=p_mn(pMax,a[1],b[1],x[1])  #? Probability at φ - dφ

    #? #* d/dφ ( <N_{s2}^{α=0}> )  
    dNs2 = -2* abs(t_s1) * abs(t_i1) * abs(t_s2)**2 * abs(V)**2 * (1+abs(V)**2)*np.sin(phi[0])
    #? Fisher Info. in the case of Singles detection (for comparisson)
    f[i]=1/(Ns2[0]*(1+Ns2[0]))*dNs2**2

p0_inv=np.zeros_like(p0)
p0_inv[p0!=0]=1/p0[p0!=0]
F = np.sum(p0_inv*((p1-pm1)/(2*dphi))**2,axis=(1,2)) #? d/dφ(p_mn) = (p_mn(φ+dφ) - p_mn(φ-dφ)) /(2φ)
fig,ax=plt.subplots()
ax.plot(phi_vec,f,color="deepskyblue",label="Singles")
ax.plot(phi_vec,F,color="royalblue",label="Coincidences")
ax.set_xticks(np.linspace(0,2*np.pi, 5))
ax.set_xticklabels([0, r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
ax.grid(True)
ax.set_xlim((0,2*np.pi))
ax.legend()
# %%
