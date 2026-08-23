#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyadd,polymul,polymulx,polyder,polyfromroots
from scipy.special import gammaln

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

    # Use gammaln rather than gamma for numerical reasons
    logCj = (gammaln(m + 1) - gammaln(m - j + 1) - 2 * gammaln(j + 1)+ j * np.log(abs(x)))
    signs = (-1)**j
    Cj[mask]= signs[mask]*np.exp(logCj[mask])
    
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
    [m,n,l]=np.indices(q.shape) #? See eq. 5.6 of thesis manuscript to understand indices 
    mask=l<=(m+n)           #? r^u_d(x)=0 whenever d>u
    q[mask]=f[l[mask]]      #? mask avoids assigns

    s=np.zeros((nMax+1,nMax+1,2*nMax+1,2*nMax+1))
    [m,n,l,j]=np.indices(s.shape) #? See eq. 5.6 of thesis manuscript to understand indices 
    mask=j<=l
    s[mask]=ra[m[mask],j[mask]]*rb[n[mask],(l-j)[mask]]
    s=np.sum(s,axis=3) #? sum over j

    return 1/(a*b) * np.sum(q*s,axis=2) #? p = sum over l


G=1                                 #? Gain
phi=np.pi/2
t_i1=np.sqrt(0.8)                     #? τ_i1
t_s1=np.sqrt(0.8)                     #? τ_s2
t_i2=np.sqrt(0.8)                     #? τ_i2  
t_s2=np.sqrt(0.8)                     #? τ_s2            

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

nMax=21
p=p_mn(nMax,a,b,x)
[i,j] = np.indices((nMax+1,nMax+1))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
xpos = i.ravel()
ypos = j.ravel()
bars = p.ravel()
colors = plt.cm.viridis((bars - bars.min()) / (bars.max() - bars.min()))
ax.bar3d(xpos,ypos,0,0.5,0.5,bars,shade=False,color=colors)
ax.set_title(rf"$p_{{mn}}$, G={G}, T={t_s1**2:.2f}, $\varphi=\pi/2$")
ax.set_xlabel("m")
ax.set_ylabel("n")
ax.yaxis.set_inverted(True)
ax.set_zlabel("p")

np.sum(p)
# %%
