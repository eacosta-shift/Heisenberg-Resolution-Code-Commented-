import numpy as np
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
    "Takes a, b, x parameters, after they are calculated beforehand"
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

def P_mn(nMax,phi=np.pi/2, G=0.5, t_s1=1, t_i1=1, t_s2=1, t_i2=1,dps=50):
    "Computes all probabilities p_mn up to m=n=nMax"
    "Takes standard parameter space as input, and computes a,b,x,|γ|^2"
    mp.dps = dps

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

    a,b,x = mpf(a), mpf(b), mpf(x)
    ra=r_all(2*nMax,1/a)
    rb=r_all(2*nMax,1/b)
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

def FisherInfo(nMax, dphi=0.001, phi=np.pi/2, G=0.5, t_s1=1, t_i1=1, t_s2=1, t_i2=1, dps=50):
    pm1 = P_mn(nMax, phi = (phi-dphi), G=G, t_s1=t_s1, t_i1=t_i1, t_s2=t_s2, t_i2=t_i2, dps=dps)    #? Probability at φ - dφ
    p0  = P_mn(nMax, phi = phi       , G=G, t_s1=t_s1, t_i1=t_i1, t_s2=t_s2, t_i2=t_i2, dps=dps)    #? Probability at φ
    p1  = P_mn(nMax, phi = (phi+dphi), G=G, t_s1=t_s1, t_i1=t_i1, t_s2=t_s2, t_i2=t_i2, dps=dps)    #? Probability at φ + dφ

    p0_inv=np.zeros_like(p0)    #? 1/p_mn
    p0_inv[p0!=0]=1/p0[p0!=0] 
    return np.sum(p0_inv*((p1-pm1)/(2*dphi))**2) #? d/dφ(p_mn) = (p_mn(φ+dφ) - p_mn(φ-dφ)) /(2φ)