import numpy as np
from scipy.interpolate import interp1d

def load_detector_psd(detector='LIGO_Design',path='src/'):
    """ 
    Loads one of the detector power spectral densities into
    a scipy interpolation function

    Available detectors are:
        'LIGO_Early'
        'LIGO_Mid'
        'LIGO_Late'
        'LIGO_Design' (default)
        'APlus'
        'ET_B'
        'ET_D'
        'CE'

    You'll need the absolute path to the src directory

    The LIGO nosie curves come from LIGO-T1800042, the
    APlus, ET_D, and CE curves from LIGO-P1600143.  The 
    ET_B files are inaccessible publicly off Virgo's website
    so I just grabbed them from Bilby
    """
    if detector == 'LIGO_Early':
        asdfile = path+'detector_files/asd.dat'
        input_freq,strain=np.loadtxt(asdfile,unpack=True,usecols=[0,1])
    elif detector == 'LIGO_Mid':
        asdfile = path+'detector_files/asd.dat'
        input_freq,strain=np.loadtxt(asdfile,unpack=True,usecols=[0,3])
    elif detector == 'LIGO_Late':
        asdfile = path+'detector_files/asd.dat'
        input_freq,strain=np.loadtxt(asdfile,unpack=True,usecols=[0,4])
    elif detector == 'LIGO_Design':
        asdfile = path+'detector_files/asd.dat'
        input_freq,strain=np.loadtxt(asdfile,unpack=True,usecols=[0,5])
    elif detector == 'APlus':
        asdfile = path+'detector_files/LIGO-T1800042-v5-aLIGO_APLUS.txt'
        input_freq,strain=np.loadtxt(asdfile,unpack=True)
    elif detector == 'ET_B':
        psdfile = path+'detector_files/ET_B_psd.txt'
        input_freq,strain=np.loadtxt(psdfile,unpack=True)
        strain = np.sqrt(strain) ## This one is PSD, not ASD
    elif detector == 'ET_D':
        asdfile = path+'detector_files/LIGO-P1600143-v18-ET_D.txt'
        input_freq,strain=np.loadtxt(asdfile,unpack=True)
    elif detector == 'CE':
        asdfile = path+'detector_files/LIGO-P1600143-v18-CE.txt'
        input_freq,strain=np.loadtxt(asdfile,unpack=True)
    else:
        print("ERROR: "+detector+" is not a valid psd file")
        return None

    psd_interp = interp1d(input_freq, strain**2)

    return psd_interp
