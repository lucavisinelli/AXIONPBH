import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.special as special
from scipy.special import gamma
from scipy.integrate import odeint
from scipy import interpolate
from scipy.interpolate import interp1d

###
### This code computes the mass of the DM axion in the PBH-dominated cosmologies
###

# Numerical constants
pi8    = 8.*np.pi

# Physical constants
mPl    = 1.221e19     # Planck mass in GeV
mgram  = 2.17651e-5   # Planck mass in grams
GN     = 1./mPl**2    # Newton's constant in GeV^-2

# PBH formation and evaporation specifications
Gpref  = 3.8          # BH graybody factor
gH     = 104.5        # Spin-weighted degrees of freedom
GgH    = Gpref*gH 
gamma  = 0.2          # See Carr ApJ 201 1 (1975)
gSM    = 106.75       # Total number of relativistic dof in the SM
g0     = 3.36         # Number of relativistic dof today

# Important measurements
rhoc0   = 8.09566e-47 # Critical density today in GeV^4
T0      = 2.34822e-13 # CMB temperature today in GeV
TBBN    = 1.e-3       # BBN temperature in GeV
z       = 0.48        # Ratio of up/down quark masses
mpi     = 0.1349768   # Neutral pion mass in GeV
fpi     = 0.092       # Neutral pion decay constant in GeV

# Derived quantities
s0      = (2.*np.pi**2/45.)*3.91*T0**3 # Entropy density today in GeV^3
Lambda4 = z/(1.+z)**2*(mpi*fpi)**2     # QCD susceptibility in GeV^4

####
####  Notation used in the code
####
####  t    : ln(T/GeV)
####  GM2  : (PBH mass over Planck mass in grams)^2
####  beta : initial fraction of PBHs at formation
####

### Interpolation for the number of relativistic dof g_star
a0 = 39.7685
a1 = np.array([  33.9935, -10.4018, -20.5193, -6.80689, -10.2647])
a2 = np.array([-891.5080,  3.81874,  -1.7175, -2.34815, -0.10613])
a3 = np.array([ 437.2980, -0.88070,  -0.1498, -1.21648, -0.94610])

def gstar(t):
    # t is ln(T/GeV)
    return a0 + sum(a1*(1.+np.tanh((t - a2) / a3)))
gstar = np.vectorize(gstar)

def gstarprime(t):
    # t is ln(T/GeV)
    # returns d gstar / d t
    return sum(a1/a3/np.cosh((t - a2) / a3)**2)
gstarprime = np.vectorize(gstarprime)

def lnHubble_rad(t):
    # t is ln(T/GeV)
    # Returns ln(Hubble rate / GeV) for the radiation-dominated Universe
    return 0.5*np.log(4*np.pi**3*GN*gstar(t)/45) + 2.*t

def Tf(GM2):
    # temperature of PBH formation in GeV
    return ((45*gamma**2)/(16*np.pi**3*GN**2*GM2*gSM))**(1/4)

def chi(t):
    # see e.g. 1610.01639, above Eq.144
    # QCD susceptibility in units of \Lambda^4
    # t is ln(T/GeV)
    return min(1., (0.157/np.exp(t))**8.16)
 
def Q(tau, GM2):
    t = tau + np.log(Tf(GM2))
    return 1./(1. + gstarprime(t)/4./gstar(t))

def tEvap(GM2):
    # time of PBH evaporation in GeV^-1 
    return (30760*np.pi*GN**(1/2)*GM2**(3/2))/GgH

def bar_rho(GM2):
    # characteristic density in GeV^4
    return 3/(pi8*GN*tEvap(GM2)**2)

def rhoR(T):
    # radiation energy density. T is in GeV
    t = np.log(T)
    return (np.pi**2/30.)*gstar(t)*T**4

def s(T):
    # Entropy density. T is in GeV
    t = np.log(T)
    return (2.*np.pi**2/45.)*gstar(t)*T**3

def y0(GM2, beta):
    # initial condition for PBH energy density
    # written as rho_BH = bar_rho* exp(y)
    rho_c = 3.*gamma**2/(32.*np.pi*GN**2*GM2)
    return np.log(beta*rho_c/bar_rho(GM2))

def taumax(GM2):
    ## End of the integration in the variable tau.
    ## returns ln(T_end / Tf), where T_end = 10^-10 T_evap
    return np.log(1.e-10*(90/(32*np.pi**3*GN*tEvap(GM2)**2*gSM))**(1/4)/Tf(GM2))

def x(tau, GM2):
    # returns x = ln (rho_R / bar_rho)
    Tf0   = Tf(GM2)
    rhoR0 = rhoR(Tf0)
    delta = np.log(rhoR0/bar_rho(GM2))
    t     = tau + np.log(Tf0)
    return 4*tau + delta + np.log(gstar(t)/gstar(np.log(Tf0)))

def my_bisection(f, a, b, tol): 
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception(
         "The scalars a and b do not bound a root")
    m = (a + b)/2
    if np.abs(f(m)) < tol:
        return m
    elif np.sign(f(a)) == np.sign(f(m)):
        return my_bisection(f, m, b, tol)
    elif np.sign(f(b)) == np.sign(f(m)):
        return my_bisection(f, a, m, tol)

def funct(Y, tau, GM2):
    y, b = Y
    xx   = x(tau, GM2)
    A    = np.sqrt(np.exp(y)+np.exp(xx))
    dydt = (1.+3.*A)/(A-np.exp(y-xx)/4.)/Q(tau, GM2)
    dbdt = -A/(A-np.exp(y-xx)/4.)/Q(tau, GM2)
    return [dydt, dbdt]

def bkgd_eqm(GM2, beta):

    ## Returns the expressions for the background
    yini = y0(GM2, beta)
    bini = 0
    Yini = [yini, bini]
    tauM = taumax(GM2)
    NT   = 10000
    ttau = np.linspace(0, tauM, NT)

    sol = odeint(funct, Yini, ttau, args=(GM2,))
    ys  = sol[:,0]       ## ln(rhoBH/rho_bar)
    bs  = sol[:,1]       ## ln(a/a0)
    xs  = x(ttau, GM2)  ## ln(rhoR/rho_bar)

    # return ln(H^2)
    H2    = np.zeros(NT)
    rhoBH = np.zeros(NT)
    for i in range(NT):
        # H2 is ln(H^2)
        H2[i]       = np.log(np.exp(ys[i])+np.exp(xs[i]))
        rhoBH[i] = np.exp(ys[i])

    ## Derivatives
    H2p = np.zeros(NT-1)
    bsp = np.zeros(NT-1)
    for i in range(NT-1):
        H2p[i] = (H2[i+1] - H2[i]) / (ttau[i+1]-ttau[i])
        bsp[i] = (bs[i+1] - bs[i]) / (ttau[i+1]-ttau[i])

    bspp = np.zeros(NT-2)
    for i in range(NT-2):
        bspp[i] = (bsp[i+1] - bsp[i]) / (ttau[i+1]-ttau[i])

    return ttau, rhoBH, H2, H2p, bs, bsp, bspp, NT

def findTosc(ma, GM2, ttau, H2):
    # 1. ma is the axion mass in eV
    # Returns the following quantities:
    # 1. tauosc = \ln(Tosc/Tf)
    # 2. Tosc  in GeV
    # 3. tevap in GeV^-1
    # 4. tauevap = \ln(Tevap/GeV)

    # tevap is the time of PBH evaporation in GeV^-1
    tauM   = taumax(GM2)
    tevap  = tEvap(GM2)
    Tini   = Tf(GM2)      # formation temperature in GeV
    # chi1 is chi(T/Tf)
    chi1   = lambda tau: chi(tau + np.log(Tf(GM2)))
    H2F  = interp1d(ttau, H2, kind='cubic', fill_value="extrapolate")

    fTosc  = lambda tau: H2F(tau) - np.log((ma*tevap/1.e9)**2*chi1(tau))
    tauosc = my_bisection(fTosc, 0, tauM, 1.e-5)
    Tosc   = Tini*np.exp(tauosc) # oscillation temperature in GeV
    findTevap = lambda t: np.log(2./(3.*tevap)) - lnHubble_rad(t)
    tauevap   = my_bisection(findTevap, 10, -7., 1.e-3)

    return tauosc, Tosc, tevap, tauevap

def axion_eqm(ma, theta_ini, GM2, beta):
    ### Solve for the axion equation
    ### xi is ln(T/Tosc)
    # solving u''+A*u'+B*sin(u) = 0
    # u is the axion angle \theta
    # v is its velocity v = u' = du/dln\xi
    # Returns the energy density at Tosc in GeV^4

    ttau, rhoBH, H2, H2p, bs, bsp, bspp, NT = bkgd_eqm(GM2, beta)
    H2F  = interp1d(ttau,         H2,   kind='cubic', fill_value="extrapolate")
    H2pF = interp1d(ttau[0:NT-1], H2p,  kind='cubic', fill_value="extrapolate")
    bF   = interp1d(ttau,         bs,   kind='cubic', fill_value="extrapolate")
    bpF  = interp1d(ttau[0:NT-1], bsp,  kind='cubic', fill_value="extrapolate")
    bppF = interp1d(ttau[1:NT-1], bspp, kind='cubic', fill_value="extrapolate")

    # express ln(H^2) and ln(a) as functions of \xi=ln(T/Tosc)
    # we write tau = xi + ln(Tosc/Tf)
    tauosc, Tosc, tevap, tauevap = findTosc(ma, GM2, ttau, H2)
    tevap_eV = tevap/1.e9
    H2n   = lambda xi: H2F  (xi+tauosc)
    H2pn  = lambda xi: H2pF (xi+tauosc)
    bn    = lambda xi: bF   (xi+tauosc)
    bpn   = lambda xi: bpF  (xi+tauosc)
    bppn  = lambda xi: bppF (xi+tauosc)
    chiQ  = lambda xi: chi  (xi+np.log(Tosc) )
    xiBBN  = np.log(TBBN/Tosc)
    xievap = tauevap - np.log(Tosc) # xievap = \ln(Tevap/Tosc)
    a3osc  = np.exp(3*bn(0))
    mosc   = ma*np.sqrt(chiQ(0))
    XI_INI =  1.0
    XI_END = -0.8
    N      = 100000
    XIS    = np.zeros(N)
    DXI    = np.zeros(N-1)
    for i in range(N):
        XIS[i] = (XI_INI - XI_END)/(N**3-1.)*i**3 + (XI_END*N**3 - XI_INI)/(N**3 - 1.)
    for i in range(N-1):
        DXI[i] = XIS[i+1]-XIS[i]

    us     = np.zeros(N)
    vs     = np.zeros(N)
    rhophi = np.zeros(N)
    us[N-1]   = theta_ini
    vs[N-1]   = 0.
    rhophi[N-1] = 0.
    for i in range(1, N):
        j  = N - i
        xi = XIS[j]
        b1 = bpn(xi)
        a3 = np.exp(3*bn(xi))
        h2 = np.exp(H2n(xi))
        mT = ma*np.sqrt(chiQ(xi))
        T1 = Tosc*np.exp(xi)
        A  = 0.5*H2pn(xi) + 3.*b1 - bppn(xi)/ b1
        B  = (mT*tevap_eV*b1)**2/h2
        u  = us[j]
        v  = vs[j]
        dxi = DXI[j-1]
        us[j-1] = u - v*dxi
        vs[j-1] = v + (A*v + B*np.sin(u))*dxi
        rhophi[j-1] = (a3/a3osc)*mosc/mT*Lambda4/(ma*tevap_eV)**2*(0.5*h2*(v/b1)**2+(mT*tevap_eV)**2*(1-np.cos(u)))

    rhomax = 0
    ximax  = 0
    imax   = 0
    for i in range(1, N):
        j  = N - i
        if rhophi[j-1] > rhomax:
            rhomax = rhophi[j-1]
            ximax  = xi
            imax   = i
        else:
            break

    rhomin = rhomax
    imin   = imax
    for i in range(imax+1, N):
        j  = N - i
        if rhophi[j-1] < rhomin:
            rhomin = rhophi[j-1]
            imin   = i
        else:
            break

    rhomax2 = rhomin
    imax2   = imin
    for i in range(imin+1, N):
        j  = N - i
        if rhophi[j-1] > rhomax2:
            rhomax2 = rhophi[j-1]
            imax2   = i
        else:
            break

    rhomin2 = rhomax2
    imin2   = imax2
    for i in range(imax2+1, N):
        j  = N - i
        if rhophi[j-1] < rhomin2:
            rhomin2 = rhophi[j-1]
            imin2   = i
        else:
            break

    avg = 0
    cnt = 0
    for i in range(imin+1, imin2+1):
        j  = N - i
        avg = avg + rhophi[j-1]
        cnt = cnt + 1
    # Energy density at Tosc in GeV^4
    rhoosc = avg / cnt
    return Tosc, mosc, rhoosc

def axion_yield(ma, GM2, ttau, rhoBH, H2, bsp, NT):
    # solving for the axion yield Y_\phi
    # here, lnY is \ln(Y_\phi)

    tauosc, Tosc, tevap, tauevap = findTosc(ma, GM2, ttau, H2)
    H2F   = interp1d(ttau, H2, kind='cubic', fill_value="extrapolate")
    bpF   = interp1d(ttau[0:NT-1], bsp, kind='cubic', fill_value="extrapolate")
    rhoMF = interp1d(ttau, rhoBH, kind='cubic', fill_value="extrapolate")
    H2n   = lambda xi: H2F  (xi+tauosc)
    bpn   = lambda xi: bpF  (xi+tauosc)
    rhM   = lambda xi: rhoMF(xi+tauosc)
    xiBBN  = np.log(TBBN/Tosc)
    gosc   = gstar(np.log(Tosc))
    xievap = tauevap - np.log(Tosc) # xievap = \ln(Tevap/Tosc)
    xi_ini = 0.0
    xi_end = xiBBN - 3.
    NXI    = 100000
    xiT    = np.zeros(NXI)
    Dxi    = np.zeros(NXI-1)
    for i in range(NXI):
        xiT[i] = (xi_ini - xi_end)/(NXI**3-1.)*i**3 + (xi_end*NXI**3 - xi_ini)/(NXI**3 - 1.)
    for i in range(NXI-1):
        Dxi[i]     = xiT[i+1]-xiT[i]
    rhobar     = 3/(pi8*GN*tevap**2)
    rhorosc    = rhoR(Tosc)
    lnYs       = np.zeros(NXI)
    lnYs[NXI-1] = 0.
    for i in range(1, NXI):
        j   = NXI - i
        xi  = xiT[j]
        b1  = bpn(xi)
        gi  = gstar(xi + np.log(Tosc))
        h   = np.sqrt(np.exp(H2n(xi)))
        lnY = lnYs[j]
        dxi = Dxi[j-1]
        A   = -0.75*b1*gosc/gi*rhobar/rhorosc*rhM(xi)*np.exp(-4.*xi)/h
        lnYs[j-1] = lnY - A*dxi

    return xi_end, lnYs[0]

def OmegaDM(ma, theta_ini, GM2, beta):
    # Returns the present axion energy density in GeV^4
    ttau, rhoBH, H2, H2p, bs, bsp, bspp, N0 = bkgd_eqm(GM2, beta)
    Tosc, mosc, rhoosc = axion_eqm(ma, theta_ini, GM2, beta)
    Yosc = rhoosc/(mosc*s(Tosc))
    xif, lnYf = axion_yield(ma, GM2, ttau, rhoBH, H2, bsp, N0)
    Yf = Yosc*np.exp(lnYf)
    Tf = Tosc*np.exp(xif)
    OmegaDM = ma*s0*Yf/rhoc0
    return OmegaDM

def axionmass(theta_ini, GM2, beta):
    # Obtain the DM axion mass with bisection
    mup = -4
    mdw = -12
    fax = lambda ma: np.log(OmegaDM(10**ma, theta_ini, GM2, beta)/0.12)
    logaxionmass = my_bisection(fax, mup, mdw, 1.e-2)
    axionmass = 10**logaxionmass
    return axionmass

plottheta = 0
if plottheta:
    ### Parameters
    beta0  = 1.e-10   # PBH fraction at formation
    MPBH   = 5.e7     # PBH mass in grams
    GM20   = (MPBH/mgram)**2
    #thetaT1 = np.pi - np.logspace(-2, np.log10(np.pi - 1.2), 10)
    #thetaT2 = np.logspace(0. ,-3., 5)
    #thetaT = np.concatenate([thetaT1, thetaT2])
    thetaT = np.logspace(np.log10(0.3) ,-3., 15)
    Ntot   = len(thetaT)
    mphiT  = np.zeros(Ntot)
    for i in range(Ntot):
        mphiT[i] = axionmass(thetaT[i], GM20, beta0)
    abs_path  = './'
    data = np.array([thetaT, mphiT])
    data = data.T
    int_file = abs_path + 'axionmass_pbh.txt'
    np.savetxt(int_file, data)

plotpbh = 0
if plotpbh:
    ### Parameters
    theta0 = 1.
    betaT  = 10**np.geomspace(-7,-14, 15)
    MPBHT  = 10**np.geomspace(6, 8.5, 15)
    GM2T   = (MPBHT/mgram)**2

    Nbeta = len(betaT)
    Nmass = len(MPBHT)
    Ntot  = Nbeta*Nmass
    mphiT  = np.zeros([Ntot, 3])
    #mphiT  = np.zeros([Nbeta, Nmass])

    for i in range(Nbeta):
        for j in range(Nmass):
            mphiT[j+Nbeta*i][0] = betaT[i]
            mphiT[j+Nbeta*i][1] = MPBHT[j]
            mphiT[j+Nbeta*i][2] = axionmass(theta0, GM2T[j], betaT[i])
    #       print(mphiT[j+Nbeta*i][2])

    #Nth    = len(thetaT)
    #mphiT  = np.zeros(Nth)
    #for i in range(Nth):
    #mphiT[i] = axionmass(thetaT[i], GM20, beta0)

    abs_path  = './'
    data = mphiT
    #data = np.array([thetaT, mphiT])
    #data = data.T
    int_file = abs_path + 'axionmass_pbh_full.txt'
    np.savetxt(int_file, data)
