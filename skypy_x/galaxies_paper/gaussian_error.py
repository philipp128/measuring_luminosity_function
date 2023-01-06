from tokenize import String
import skypy.utils.photometry as phot
import sys
sys.path.insert(1, '../')
#from rykoff_flux_error import flux_error_rykoff
import numpy as np

def gaussian_error_magnitude(m, m_lim, rykoff_parameters, m_zp, error_limit=np.inf):
    # rykoff parameters[:, 0] =  array of a for each band
    # rykoff parameters[:, 1] = array of b for each band
    # rykoff parameters can also be file_path as string but the syntax must be as above
    if isinstance(rykoff_parameters, str):
        rykoff_parameters = np.loadtxt(rykoff_parameters)
    
    sigma = phot.magnitude_error_rykoff(m, m_lim, m_zp, rykoff_parameters[:, 0], 
                                        rykoff_parameters[:, 1], error_limit)

    return np.random.normal(loc=0, scale=sigma), sigma

def gaussian_error_flux(m, m_lim, rykoff_parameters, m_zp, error_limit=np.inf):
    # rykoff parameters[:, 0] =  array of a for each band
    # rykoff parameters[:, 1] = array of b for each band
    # rykoff parameters can also be file_path as string but the syntax must be as above
    if isinstance(rykoff_parameters, str):
        rykoff_parameters = np.loadtxt(rykoff_parameters)
    
    sigma = flux_error_rykoff(m, m_lim, m_zp, rykoff_parameters[:, 0], 
                              rykoff_parameters[:, 1], error_limit)

    return np.random.normal(loc=0, scale=sigma), sigma
