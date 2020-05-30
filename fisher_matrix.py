import numpy as np
from useful import gravywaves as gw, cosmo as co
from astropy import units as u, constants as c
import lal
import lalsimulation as ls
import pandas as pd
from scipy.optimize import leastsq,brentq
from scipy.interpolate import interp1d
from scipy.integrate import quad

def add_default_units(quant,unit):
    if type(quant) == u.quantity.Quantity:
        return quant
    else:
        return quant*unit

## Rather than recompute the the frequency and angular variables,
## just import the table from Emanuele's website

## Yes, yes, it's poor form to have global variables, 
## but honestly it's so much eaasier here
kerr_qnm = pd.read_pickle('Kerr_QNM_coefs/PandaBerti.pkl')

def spin_index(a):
    """
    Get the spin index for the dataframe
    """
    return int(np.round(a,4)*1e4)

def mode_index(n,l,m):
    """
    Convert n,l,m tuple into the correct column of the dataframe

    Note that you can also just access it as, e.g., `n2_l2_m0_ImAlm`
    for the n=2,l=2,m=0 imaginary part of the amplitude
    """
    return 4*(60*(n-1)+(l**2 + l + m - 4))

def get_qnm(a,n,l,m):
    """
    Looks up the coefficients of the Kerr quasinormal modes
    from Emanuele's tables

    Returns omega, A (complex numbers)
    """
    reO,imO,reA,imA = kerr_qnm.iloc[spin_index(a),mode_index(n,l,m):(mode_index(n,l,m)+4)]

    return reO+imO*1j, reA+imA*1j

def recurrence_coefficients(l, m, nn, aOmega, A_nlm):
    """
    Compute the angular expansion coeffeicients for the Kerr modes
    Using Equation 20 of Leaver 1985

    Takes in the spin, l, m, k, Alm, Omegalm
    return (alpha, beta, gamma) (complex numbers)

    NOTE: a is from -0.5 to 0.5, because Leaver uses C=G=2M=1.
        This also means omega is times 2...

    ALSO NOTE: nn is for the sum in the spherical function, NOT the overtone
    """
    s = -2

    k1 = abs(m - s) / 2
    k2 = abs(m + s) / 2

    alpha = -2.0*(nn + 1.0)*(nn + 2.0*k1 + 1.0)
    beta = (nn*(nn - 1.0) + 2.0*nn*(k1+k2+ 1.0 - 2.0 * aOmega)
            - (2.0*aOmega * (2.0*k1 + s + 1.0) - (k1 + k2) * (k1 + k2 + 1.0))
            - (aOmega**2 + s*(s+1.0) + A_nlm))
    gamma = 2*aOmega*(nn + k1 + k2 + s)

    return (alpha,beta,gamma)

def spherical_lm_unnormalized(u, l, m, aOmega, A_nlm):
    """
    Computes the spheroidal wavefunctions (s=-2) from equations
    18, 19, and 20 of Leaver 1985

    NOTE: a is from -0.5 to 0.5, because Leaver uses C=G=2M=1.
        This also means omega is times 2...
    """

    s = -2

    k1 = abs(m - s) / 2
    k2 = abs(m + s) / 2

    sum_N = 500
    a_n = np.zeros(sum_N, dtype=np.complex128)

    ## First set the initial a_n[N-1] and a_n[N]
    alpha,beta,gamma = recurrence_coefficients(l, m, 0, aOmega, A_nlm)
    a_n[0] = 1
    a_n[1] = -beta/alpha

    ## Then do the recurrence part
    nn = 1
    while nn < sum_N-1:
        alpha,beta,gamma = recurrence_coefficients(l, m, nn, aOmega, A_nlm)
        a_n[nn+1] = -(beta*a_n[nn] + gamma*a_n[nn-1]) / alpha

        nn += 1

    ## compute the sum in eqn. 18
    sum_a_n = np.sum(a_n*((1+u)**np.arange(0,sum_N)))

    ## finally the coefficient in 18
    coef = np.exp(aOmega*u) * (1+u)**k1 * (1-u)**k2

    return coef*sum_a_n


def spherical_lm(u, a, l, m, Omega_nlm, A_nlm):
    """
    Computes the spheroidal wavefunctions (s=-2) from equations
    18, 19, and 20 of Leaver 1985, then normalizes them

    NOTE: Leaver uses c=G=2M=1, but since only a*omega enters the equations, the 
    2's cancel
    """
    aOmega = a*Omega_nlm

    integrand = lambda x: np.abs(spherical_lm_unnormalized(x, l, m, aOmega, A_nlm))**2

    norm,_ = quad(integrand,-1,1)

    return spherical_lm_unnormalized(u, l, m, aOmega, A_nlm) / np.sqrt(norm)


def hp_hx_ringdown_time_domain(m_final, a_final, m_frac, inclination, dist, 
        phi0, delta_T, T_max ,n=1, l=2, m=2):

    ## add default units, then convert them to seconds because relativity
    m_final = add_default_units(m_final, u.solMass)
    dist    = add_default_units(dist, u.Mpc)

    m_final  = (m_final * c.G / c.c**3).to(u.s).value
    dist     = (dist / c.c).to(u.s).value
    m_over_r = m_final/dist

    times = np.arange(0,T_max,delta_T)

    ##  First get the QNM coefficients and compute the frequency/quality factor
    Omega_nlm, A_nlm = get_qnm(a_final,n,l,m)

    w_nlm = Omega_nlm/m_final
    tau_nlm = 1 / abs(w_nlm.imag) 
    one_over_tau_nlm = 1. / tau_nlm
    f_nlm = w_nlm.real / (2 * np.pi) 
    Q_nlm = np.pi*f_nlm*tau_nlm 

    ## calculate this with the spherical functions
    mu = np.cos(inclination)
    phase = np.exp(1j*m*phi0)
    s_nlm      = spherical_lm(mu, a_final, l, m, Omega_nlm, A_nlm)*phase
    s_nlm_conj = np.conj(spherical_lm(-mu, a_final, l, m, 
        Omega_nlm,A_nlm)*phase)


    ## Folllowing 3.6 of Berti et al.
    h = (np.exp(1j*(2*np.pi*f_nlm*times) - times/tau_nlm)*s_nlm + 
            np.exp(1j*(-2*np.pi*f_nlm*times) - times/tau_nlm)*s_nlm_conj)

    ## 3.18 of same
    amp = np.sqrt(32*Q_nlm*m_frac / (f_nlm*(1+4*Q_nlm**2)))

    return m_over_r*amp*h.real, m_over_r*amp*h.imag, times

def hp_hx_ringdown(m_final, a_final, m_frac, inclination, dist, phi0, 
        df=0.1, fmin=5,fmax=2000,n=1, l=2, m=2):

    ## add default units, then convert them to seconds because relativity
    m_final = add_default_units(m_final, u.solMass)
    dist    = add_default_units(dist, u.Mpc)

    m_final  = (m_final * c.G / c.c**3).to(u.s).value
    dist     = (dist / c.c).to(u.s).value
    m_over_r = m_final/dist

    freqs = np.arange(fmin,fmax,df)

    ##  First get the QNM coefficients and compute the frequency/quality factor
    Omega_nlm, A_nlm = get_qnm(a_final,n,l,m)

    w_nlm = Omega_nlm/m_final
    tau_nlm = 1 / abs(w_nlm.imag) 
    one_over_tau_nlm = 1. / tau_nlm
    f_nlm = w_nlm.real / (2 * np.pi) 
    Q_nlm = np.pi*f_nlm*tau_nlm 

    b_plus  = tau_nlm / (1 + (2*np.pi*tau_nlm*(freqs + f_nlm))**2)
    b_minus = tau_nlm / (1 + (2*np.pi*tau_nlm*(freqs - f_nlm))**2)

    ## calculate this with the spherical functions
    mu = np.cos(inclination)
    phase = np.exp(1j*m*phi0)
    s_nlm     = spherical_lm(mu, a_final, l, m, Omega_nlm, A_nlm)*phase
    s_nlm_mmu = spherical_lm(-mu, a_final, l, m, Omega_nlm,A_nlm)*phase

    ## ok now code up the actual waveform.  Note that we explicitly assume 
    ## (ala Flanagan and Hughes 1998 and Berti et al., 2006) that 
    ## phi^x = phi^+ = 0, and that A^+ = A^x = A
    ##
    ## The fourier transform of the time-domain waveform; note that this 
    ## needs to be done carefully to keep track of the S_nlm and S_nlm* 
    ## components

    hp = b_plus*np.conj(s_nlm+s_nlm_mmu) + b_minus*(s_nlm+s_nlm_mmu)
    hx = 1j*(b_plus*(s_nlm_mmu-s_nlm) + b_minus*np.conj(s_nlm-s_nlm_mmu))

    ## 3.18 of Berti et al., 2006 
    amp = np.sqrt(32*Q_nlm*m_frac / (f_nlm*(1+4*Q_nlm**2)))

    return m_over_r*amp*hp, m_over_r*amp*hx,freqs


def hp_hx_inspiral(m1,m2,dist,phase=0.,df=1e-2,
                s1x=0.0,s1y=0.0,s1z=0.0,s2x=0.0,s2y=0.0,s2z=0.0,
                fmin=1.,fmax=0.,fref=1.,iota=0.,longAscNodes=0.,
                eccentricity=0.,meanPerAno=0.,LALpars=None,
                approx=ls.IMRPhenomD):
    """
    Wrapper for a LAL waveform, courtesy of Chris Pankow
    m1,m2 in solar masses, distance in parsec
    
    returns (hplus, hcross, frequency)
    """

    hplus_tilda, hcross_tilda = ls.SimInspiralChooseFDWaveform(m1*lal.MSUN_SI, m2*lal.MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z, 
                                                        dist*(1E6 * ls.lal.PC_SI),iota,phase,longAscNodes,eccentricity,meanPerAno,df,
                                                        fmin,fmax,fref,LALpars,approx)

    freqs=np.array([hplus_tilda.f0+i*hplus_tilda.deltaF for i in np.arange(hplus_tilda.data.length)])

    return hplus_tilda.data.data,hcross_tilda.data.data,freqs

def pattern_functions(theta,phi,psi,triangle=False):
    """
    The pattern functions for an L-shapped detector

    The triangle flag multiplies the whole result by sqrt(3)/2, for 
    detectors that are triangle shapped (ET, LISA)

    Returns F+, Fx
    """

    ctheta = np.cos(theta)
    ctheta_squ = ctheta**2
    c2phi = np.cos(2*phi)
    s2phi = np.sin(2*phi)
    c2psi = np.cos(2*psi)
    s2psi = np.sin(2*psi)

    F_plus  = 0.5*(1+ctheta_2)*c2phi*c2psi - ctheta*s2phi*s2psi
    F_cross = 0.5*(1+ctheta_2)*c2phi*s2psi + ctheta*s2phi*c2psi

    if triangle:
        F_plus  *= 0.8660254
        F_cross *= 0.8660254

    return F_plus,F_cross
