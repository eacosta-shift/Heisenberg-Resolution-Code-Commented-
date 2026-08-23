#%%
#%matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, mpf, polyval, exp, log, loggamma

def polyadd(p, q):
    "Polynomial Sum (only coeffs.)"
    n,m=len(p),len(q)
    length=max(n, m)
    p=[mpf(0)]*(length-n) + p 
    q=[mpf(0)]*(length-m) + q 
    return [pi+qi for pi,qi in zip(p,q)]

def polymul(p, q):
    "Polynomial Multiplication (only coeffs.)"
    result=[mpf(0)]*(len(p)+len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            result[i+j]+=a*b
    return result

def polyder(p):
    "Polynomial Derivative (only coeffs.)"
    return [(len(p)-(1+i)) * p[i] for i in range(0, len(p)-1)]

def pochhammer(m):
    "Pochhammer symbol"
    p=[mpf(1)]
    for i in range(m):
        p = polymul(p,[1,mpf(i+1)])
    return p
        
def f_all(oMax, x_val, dps=50):
    "Computes all the function f_n(x) up to n = oMax."
    mp.dps = dps  # set decimal precision
    x = mpf(x_val)
    f = [mpf(1) / (1 - x)]  # f[0]
    Q = [[mpf(1)]]  # Q[0] = [1]

    for n in range(1, oMax + 1):
        term1 = polymul([-mpf(1), mpf(1)], polyder(Q[n-1])) 
        term2 = [n*c for c in Q[n-1]]                      
        Qn = polymul([mpf(1), mpf(0)], polyadd(term1, term2)) 
        Q.append(Qn)
        f.append(polyval(Qn, x) / (1 - x) ** (n + 1))

    return f

def r_all(oMax,x,dps=50):
    "Compute all r^m_j(x) up to m=j=oMax."
    "Use r[m][j] to call an element of r^m_j(x)]"

    mp.dps = dps
    x = mpf(x)
    r = [[mpf(0) for _ in range(oMax+1)] for _ in range(oMax+1)]

    for m in range(oMax+1):
        hyper_m=[mpf(0)]
        for j in range(m+1):
            sign=(-1)**j
            logCj=loggamma(m+1)-loggamma(m-j+1)-2*loggamma(j + 1)+j*log(abs(x))
            Cj=sign*exp(logCj)
            poly_k=pochhammer(j)
            poly_k= [Cj*i for i in poly_k] 
            hyper_m=polyadd(hyper_m,poly_k)
        r[m][0:m+1]=hyper_m[::-1]
    return r

def p_mn(nMax,a,b,x,dps=50):
    "Computes all probabilities p_mn up to m=n=nMax"
    mp.dps = dps
    a,b,x = mpf(a), mpf(b), mpf(x)
    ra=r_all(2*nMax,1/a, dps=dps)
    rb=r_all(2*nMax,1/b, dps=dps)
    f=f_all(2*nMax,x)

    p = [[mpf(0) for _ in range(nMax + 1)] for _ in range(nMax + 1)]

    for m in range(nMax+1):
        for n in range(nMax+1):
            pmn=mpf(0)
            for l in range(m+n+1):
                inner_sum=mpf(0)
                for j in range(l+1):
                    inner_sum+=ra[m][j]*rb[n][l-j]
                pmn+=inner_sum*f[l]/(a*b)
            p[m][n]=pmn

    return np.array(p,dtype=float)
    

G=1                           #? Gain
phi=3*np.pi/4                 #? φ
t_i1=np.sqrt(0.8)             #? τ_i1
t_s1=np.sqrt(1)               #? τ_s2
t_i2=t_i1                     #? τ_i2  
t_s2=t_s1                     #? τ_s2            

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


nMax=25
p=p_mn(nMax,a,b,x,dps=50)
[i,j] = np.indices((nMax+1,nMax+1))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
xpos = i.ravel()
ypos = j.ravel()
bars = p.ravel()
colors = plt.cm.viridis((bars - bars.min()) / (bars.max() - bars.min()))
ax.bar3d(xpos, ypos, 0, 0.5, 0.5, bars,
         shade=False, color=colors, zsort='max')

ax.set_xlabel("m")
ax.set_ylabel("n")
ax.set_xlim(0,nMax+1)
ax.set_ylim(0,nMax+1)
ax.yaxis.set_inverted(True)
ax.set_xticks(np.arange(0, nMax+1,5))
ax.set_yticks(np.arange(0, nMax+1,5))
ax.set_zlim3d((0,0.4))
ax.grid(True)

ax.xaxis.pane.set_facecolor('white')
ax.yaxis.pane.set_facecolor('white')
ax.zaxis.pane.set_facecolor('white')

np.sum(p)
# %%
