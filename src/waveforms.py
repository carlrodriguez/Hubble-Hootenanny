import numpy as np
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
try:
    kerr_qnm = pd.read_pickle('PandaBerti.pkl')
except FileNotFoundError:
    print("You don't have the Kerr QNM table locally.")
    print("Use the build_kerr_qnm_table function.")

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

def spherical_lm(u, a, l, m, Omega_nlm, A_nlm):
    """
    Computes the spheroidal wavefunctions (s=-2) from equations
    18, 19, and 20 of Leaver 1985, then normalizes them

    NOTE: Leaver uses c=G=2M=1, but since only a*omega enters the equations, the 
    2's cancel
    """
    s = -2

    k1 = abs(m - s) / 2
    k2 = abs(m + s) / 2

    aOmega = a*Omega_nlm

    sum_N = 100

    ## Computing the coefficients for the recurrence relation can be done
    ## outside the main part of th s_nlm
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

    ## now integrate and normalize
    integrand = lambda x: np.abs(np.exp(aOmega*x) * (1+x)**k1 * (1-x)**k2 * 
                            np.sum(a_n*((1+x)**np.arange(0,sum_N))))**2

    norm,_ = quad(integrand,-1,1)

    return np.sqrt(integrand(u) / norm)


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

    ## Amplitude from Baibhav and Berti 2018 (1809.03500)
    amp = np.sqrt(4*m_frac/(m_final*Q_nlm*f_nlm))

    return m_over_r*amp*h.real, m_over_r*amp*h.imag, times

def hp_hx_ringdown_single_mode(m_final, a_final, m_frac, inclination, dist, phi0, 
        freqs,n=1, l=2, m=2):

    ## add default units, then convert them to seconds because relativity
    m_final = add_default_units(m_final, u.solMass)
    dist    = add_default_units(dist, u.Mpc)

    m_final  = (m_final * c.G / c.c**3).to(u.s).value
    dist     = (dist / c.c).to(u.s).value
    m_over_r = m_final/dist

    ##  First get the QNM coefficients and compute the frequency/quality factor
    Omega_nlm, A_nlm = get_qnm(a_final,n,l,m)

    w_nlm = Omega_nlm/m_final
    tau_nlm = 1 / abs(w_nlm.imag) 
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
    hx = -1j*(b_plus*(s_nlm_mmu+s_nlm) - b_minus*np.conj(s_nlm-s_nlm_mmu))

    #hp = b_plus*s_nlm + b_minus*np.conj(s_nlm_mmu)
    #hx = -1j*(b_plus*s_nlm - b_minus*np.conj(s_nlm_mmu))

    ## Amplitude from Baibhav and Berti 2018 (1809.03500)
    amp = np.sqrt(16*Q_nlm*m_frac / (m_final*f_nlm*(1+4*Q_nlm**2)))
    ## The 2*pi comes from the fourier transform, and the other 2 from 
    ## The Flanagan and Hughes way of the FT (assuming the ringdown is 
    ## symmetric about 0, then divide by sqrt2 to compensate)
    amp /= np.sqrt(4*np.pi)

    return m_over_r*amp*hp, m_over_r*amp*hx

def hp_hx_ringdown(m_final=100, a_final=0.69, q=1, chi_p=0, chi_m=0, inclination=0, dist=500, phi0=0, 
        df=0.1, fmin=5,fmax=2000, l=None, m=None):
    """
    Computes the ringdown waveform in the frequency domain.  Uses the amplitudes 
    from Baibhav and Berti 2018 (1809.03500) as a function of chi_p 
    (chi_effective), chi_m ((m1*sz1 - m2*sz2)/(m1+m2)) and mass ratio q (greater 
    than 1).  

    By default, l=None, m=None includes the (2,2), (3,3), (4,4), and (2,1) modes
    with their appropriate amplitudes.  If you select one of those individual 
    nodes, it will return just that one (with appropriate amplitude).  

    Note I'm assuming equal amplitudes in h+ and hx, and that all initial phases 
    for the ringdowns (except for the phi0 of the source in the spheriodal harmonics) 
    are identically 0.

    returns hp, hx, frequencies
    """
    
    if q < 1.:
        q = 1./q

    eta = q / (1+q)**2
    delta = (q - 1) / (q + 1)

    A0_22 = 0.303 + 0.571*eta
    A0_33 = 0.157 + 0.671*eta
    A0_21 = 0.099 + 0.06*eta
    A0_44 = 0.122 - 0.188*eta - 0.964*eta*eta

    AS_22 = eta*chi_p*(-0.07 + 0.255/q + 0.189*q - 0.013*q*q) + 0.084*delta*chi_m
    AS_33 = eta*chi_m*(0.163 - 0.187/q + 0.021*q) + 0.073*delta*chi_p
    AS_21 = -0.067*chi_m
    AS_44 = eta*chi_p*(-0.207/q + 0.034*q) + delta*eta*chi_m*(-0.701 + 1.387/q + 0.122*q)

    E_22 = (eta*(A0_22+AS_22))**2
    E_44 = (eta*(A0_44+AS_44))**2
    E_21 = (eta*(np.sqrt(1-4*eta)*A0_21 + AS_21))**2
    E_33 = (eta*(np.sqrt(1-4*eta)*A0_33 + AS_33))**2

    freqs = np.arange(fmin,fmax,df)


    if l == None and m == None:
        hp_22,hx_22 = hp_hx_ringdown_single_mode(m_final, a_final, E_22, inclination, dist, phi0, freqs, l=2, m=2)
        hp_44,hx_44 = hp_hx_ringdown_single_mode(m_final, a_final, E_44, inclination, dist, phi0, freqs, l=4, m=4)
        hp_21,hx_21 = hp_hx_ringdown_single_mode(m_final, a_final, E_21, inclination, dist, phi0, freqs, l=2, m=1)
        hp_33,hx_33 = hp_hx_ringdown_single_mode(m_final, a_final, E_33, inclination, dist, phi0, freqs, l=3, m=3)
        return hp_22+hp_44+hp_21+hp_33, hx_22+hx_44+hx_21+hx_33, freqs
    elif l == 2 and m == 2:
        hp_22,hx_22 = hp_hx_ringdown_single_mode(m_final, a_final, E_22, inclination, dist, phi0, freqs, l=2, m=2)
        return hp_22,hx_22,freqs
    elif l == 2 and m == 1:
        hp_21,hx_21 = hp_hx_ringdown_single_mode(m_final, a_final, E_21, inclination, dist, phi0, freqs, l=2, m=1)
        return hp_21,hx_21,freqs
    elif l == 4 and m == 4:
        hp_44,hx_44 = hp_hx_ringdown_single_mode(m_final, a_final, E_44, inclination, dist, phi0, freqs, l=4, m=4)
        return hp_44,hx_44,freqs
    elif l == 3 and m == 3:
        hp_33,hx_33 = hp_hx_ringdown_single_mode(m_final, a_final, E_33, 
                inclination, dist, phi0, freqs, l=3, m=3)
        return hp_33,hx_33,freqs
    else:
        print("ERROR: this (l,m) is not one we have amplitude corrections for")
        return 0,0,0



def hp_hx_inspiral(mtotal=100,q=1,dist=500,phi0=0.,df=1e-2,
                s1x=0.0,s1y=0.0,chi_p=0.0,s2x=0.0,s2y=0.0,chi_m=0.0,
                fmin=5.,fmax=2000.,fref=1.,inclination=0.,longAscNodes=0.,
                eccentricity=0.,meanPerAno=0.,LALpars=None,
                approx=ls.IMRPhenomPv3HM):
    """
    Wrapper for a LAL waveform, courtesy of Chris Pankow
    mtotal in solar masses, mass ratio q > 1,  distance in parsec
    
    returns (hplus, hcross, frequency)
    """

    if q < 1:
        q = 1/q

    m2 = mtotal / (1+q)
    m1 = mtotal / (1+1/q)
    s1z = (m1+m2)*(chi_p+chi_m) / (2*m1)
    s2z = (m1+m2)*(chi_p-chi_m) / (2*m2)

    hplus_tilda, hcross_tilda = ls.SimInspiralChooseFDWaveform(m1*lal.MSUN_SI, m2*lal.MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z, 
                                                        dist*(1E6*ls.lal.PC_SI),inclination,phi0,longAscNodes,eccentricity,meanPerAno,df,
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

    F_plus  = 0.5*(1+ctheta_squ)*c2phi*c2psi - ctheta*s2phi*s2psi
    F_cross = 0.5*(1+ctheta_squ)*c2phi*s2psi + ctheta*s2phi*c2psi

    if triangle:
        F_plus  *= 0.8660254
        F_cross *= 0.8660254

    return F_plus,F_cross

def build_kerr_qnm_table(folder):
    """
    Function to build a Pandas file of all the Kerr QNM coefficients
    from Berti et al., 2006.

    It takes as input a path to a folder where you have all of the 
    unzipped files from https://pages.jh.edu/~eberti2/ringdown/ 
    unzipped (in this case, l2-l7 for the s = -2 case)

    Saves the file as a pandas pickle in the current folder
    """
    ## The script to put all of the Kerr QNMs from
    ## Berti et al. 2006 into a pandas dataframe

    FirstFile = True

    for n in range(1,9):
        for l in range(2,8):
            for m in range(-l,l+1):

                if m < 0:
                    m_str = 'm'+str(-m)
                else:
                    m_str = str(m)

                filename = folder+'/l'+str(l)+'/n'+str(n)+'l'+str(l)+'m'+m_str+'.dat'
                col_1 = 'n'+str(n)+'_l'+str(l)+'_m'+str(m)+'_ReOmega'
                col_2 = 'n'+str(n)+'_l'+str(l)+'_m'+str(m)+'_ImOmega'
                col_3 = 'n'+str(n)+'_l'+str(l)+'_m'+str(m)+'_ReAlm'
                col_4 = 'n'+str(n)+'_l'+str(l)+'_m'+str(m)+'_ImAlm'

                column_list = [col_1, col_2, col_3, col_4]

                df = pd.read_csv(filename,sep=' ',index_col=0,header=None)
                df.columns = column_list

                if FirstFile:
                    df_full = df
                    FirstFile = False
                else:
                    for column in column_list:
                        df_full[column] = pd.Series(df[column])

    df_full.to_pickle("PandaBerti.pkl")
