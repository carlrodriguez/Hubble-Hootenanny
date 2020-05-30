import numpy as np
from astropy import units as u, constants as c

#take the h+ component to calculate SNR
def compute_SNR(hplus_tilda,psd_interp,fsel,df):
    return sqrt(4.*df*sum(abs(hplus_tilda[fsel])**2/psd_interp)) 
