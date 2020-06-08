import numpy as np
from scipy.integrate import trapz
from astropy import units as u, constants as c

def inner_product(h1,h2,freqs,psd_interp):
    """
    Compute the one-sided inner production of two frequency-domain 
    gravitational waves

    i.e. 2 * \int_0^\inf (h1*conj(h2) + conj(h1)*h2)/S_h(f) df

    input is two gravitational waves evaluated at frequencies freq, and
    a scipy interpolator for the psd
    """

    ## Make sure we don't go beyond the range of the PSD
    f_min, f_max = psd_interp.x[0], psd_interp.x[-1]
    valid_freqs = (freqs > f_min) & (freqs < f_max)

    sum_prod = (h1*np.conj(h2) + np.conj(h1)*h2)

    integral = trapz(sum_prod[valid_freqs] / psd_interp(freqs[valid_freqs]), freqs[valid_freqs])

    return 2*integral.real ##It's always real, but this retypes it as a double 

def compute_SNR(h,freqs,psd_interp):
    """
    Compute the SNR of h, evaluated at frequencies freqs with psd_interp
    given as a scipy interpolator
    """
    return np.sqrt(inner_product(h,h,freqs,psd_interp))

def function_derivatives(h_func,deriv_key,params,delta_xs):
    """
    Computes the derivative of the gravitational wave given by 5-point 

    h_func is the function, 
    deriv_key is the key for the parameter we want to differentiate with respect to
    params is the dictionary of waveform parameters
    delta_xs is the dictionary of derivative step sizes

    returns hp, hx, freqs
    """

    param_center = params[deriv_key]
    h = delta_xs[deriv_key]

    params[deriv_key] = param_center + 2*h
    hp_p2,hx_p2,_ = h_func(**params)

    params[deriv_key] = param_center + h 
    hp_p1,hx_p1,_ = h_func(**params)

    params[deriv_key] = param_center - h 
    hp_m1,hx_m1,_ = h_func(**params)

    params[deriv_key] = param_center - 2*h
    hp_m2,hx_m2,freqs = h_func(**params)

    hp = (-hp_p2 + 8*hp_p1 - 8*hp_m1 + hp_m2) / (12*h)
    hx = (-hx_p2 + 8*hx_p1 - 8*hx_m1 + hx_m2) / (12*h)

    ## Dictionaries are mutable in python, so put this back where it was
    params[deriv_key] = param_center

    return hp, hx, freqs


    

     
