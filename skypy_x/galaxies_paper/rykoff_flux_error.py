import numpy as np
import skypy.utils.photometry as phot

def flux_error_rykoff(magnitude, magnitude_limit, magnitude_zp, a, b, error_limit=np.inf):
    """ Calculate Rykoff flux error.

        Eq. (5) from Rykoff E. S., Rozo E., Keisler R., 2015, eprint arXiv:1509.00870
    """
    flux = phot.luminosity_from_absolute_magnitude(magnitude, -magnitude_zp)
    flux_limit = phot.luminosity_from_absolute_magnitude(magnitude_limit, -magnitude_zp)
    t_eff = np.exp(a + b * np.subtract(magnitude_limit, 21.0))
    flux_noise = np.square(flux_limit / 10) * t_eff - flux_limit
    error = np.sqrt((flux + flux_noise) / t_eff)
    
    return np.minimum(error, error_limit)
