import sys
sys.path.insert(1, '../')
from gaussian_error import gaussian_error_magnitude, gaussian_error_flux
import numpy as np

def test_gaussian_error_mgnitude():
    m = np.full((2, 1, 1, 1), 21)
    m_lim = np.full((3, 1, 1), 21)
    m_zp = np.full((5, 1), 21)

    rykoff_parameters = np.array([
        [1.541275, -1.000737],
        [135.095407, 20.425863],
        [3.41, 1.15]
    ])

    m_error, sigma = gaussian_error_magnitude(m, m_lim, rykoff_parameters, m_zp)

    assert m_error.shape == (2, 3, 5, 3)
    assert sigma.shape == (2, 3, 5, 3)

    parameter_file = 'test_data/rykoff_parameter.txt'
    rykoff_parameters = np.loadtxt(parameter_file)

    m_error, sigma = gaussian_error_magnitude(m, m_lim, parameter_file, m_zp)

    assert np.allclose(sigma, 0.25 / np.log(10))

def test_gaussian_error_flux():
    m = np.full((2, 1, 1, 1), 21)
    m_lim = np.full((3, 1, 1), 21)
    m_zp = np.full((5, 1), 21)

    rykoff_parameters = np.array([
        [1.541275, -1.000737],
        [135.095407, 20.425863],
        [3.41, 1.15]
    ])

    m_error, sigma = gaussian_error_flux(m, m_lim, rykoff_parameters, m_zp)

    assert m_error.shape == (2, 3, 5, 3)
    assert sigma.shape == (2, 3, 5, 3)

    parameter_file = 'test_data/rykoff_parameter.txt'
    rykoff_parameters = np.loadtxt(parameter_file)

    m_error, sigma = gaussian_error_flux(m, m_lim, parameter_file, m_zp)

    assert np.allclose(sigma, 1.0 / 10)
    

