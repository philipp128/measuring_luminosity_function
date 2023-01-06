import sys
sys.path.insert(1, '../')
from rykoff_flux_error import flux_error_rykoff
import numpy as np

def test_rykoff_flux_error():
    # Test broadcasting to same shape given array for each parameter and
    # test for correct result.
    magnitude = np.full((2, 1, 1, 1, 1), 21)
    magnitude_limit = np.full((3, 1, 1, 1), 21)
    magnitude_zp = np.full((5, 1, 1), 21)
    a = np.full((7, 1), np.log(200))
    b = np.zeros(11)
    error = flux_error_rykoff(magnitude, magnitude_limit, magnitude_zp, a, b)
    # test result
    assert np.allclose(error, 1.0 / 10)
    # test shape
    assert error.shape == (2, 3, 5, 7, 11)

    # second test for result
    magnitude = 20
    magnitude_limit = 22.5
    magnitude_zp = 25
    b = 2
    a = np.log(100) - 1.5 * b
    error = flux_error_rykoff(magnitude, magnitude_limit, magnitude_zp, a, b)
    assert np.isclose(error, np.sqrt(190/100))

    # test that error limit is returned if error is larger than error_limit
    # The following set-up would give a value larger than 1.37
    magnitude = 20
    magnitude_limit = 22.5
    magnitude_zp = 25
    b = 2
    a = np.log(100) - 1.5 * b
    error_limit = 1
    error = flux_error_rykoff(magnitude, magnitude_limit, magnitude_zp, a, b, error_limit)
    assert error == error_limit
