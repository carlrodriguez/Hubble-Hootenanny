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

def function_derivatives(h_func,deriv_key,delta_h,params):
    """
    Computes the derivative of the gravitational wave given by 5-point 

    h_func is the function, 
    deriv_key is the key for the parameter we want to differentiate with respect to
    params is the dictionary of waveform parameters
    delta_h is the derivative step size

    returns h_deriv, freqs
    """

    param_center = params[deriv_key]

    params[deriv_key] = param_center + 2*delta_h
    h_p2,_ = h_func(**params)

    params[deriv_key] = param_center + delta_h 
    h_p1,_ = h_func(**params)

    params[deriv_key] = param_center - delta_h 
    h_m1,_ = h_func(**params)

    params[deriv_key] = param_center - 2*delta_h
    h_m2,freqs = h_func(**params)

    h_der = (-h_p2 + 8*h_p1 - 8*h_m1 + h_m2) / (12*delta_h)

    ## Dictionaries are mutable in python, so put this back where it was
    params[deriv_key] = param_center

    return h_der, freqs

def fisher_matrix(h_func,deriv_keys,params,delta_xs,psd_interp):
    
    num_params = len(deriv_keys)
    fm = np.zeros(shape=(num_params,num_params))

    ## first precompute the derivatives
    dh = {}
    for deriv_key in deriv_keys:
        delta_h = delta_xs[deriv_key]
        h_der,freqs = function_derivatives(h_func,deriv_key,delta_h,params) 
        dh[deriv_key] = h_der 

    ## Now evaluate the actual Fisher Matrix
    for i,i_name in enumerate(deriv_keys):
        for j,j_name in enumerate(deriv_keys[i:]):
            fm[i,j+i] = inner_product(dh[i_name],dh[j_name],freqs,psd_interp)
            fm[j+i,i] = fm[i,j+i]

    return fm
