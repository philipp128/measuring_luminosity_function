import pytest


def test_uvista_templates():
    import numpy as np
    from skypy_x.galaxies_paper import uvista

    assert uvista.num_templates == 12

    assert uvista.wavelength.size > 0
    assert uvista.templates.shape == (uvista.num_templates, uvista.wavelength.size)


def test_uvista_magnitudes():
    import numpy as np
    from astropy.cosmology import Planck15
    from skypy_x.galaxies_paper import uvista

    # Test returned array shapes with single and multiple filters
    ng, nt = 7, 12
    coeff = np.ones((ng, nt))
    multiple_filters = ['decam2014-g', 'decam2014-r']
    nf = len(multiple_filters)
    z = np.linspace(1, 2, ng)

    MB = uvista.absolute_magnitudes(coeff, 'bessell-B')
    assert np.shape(MB) == (ng,)

    MB = uvista.absolute_magnitudes(coeff, multiple_filters)
    assert np.shape(MB) == (ng, nf)

    mB = uvista.apparent_magnitudes(coeff, z, 'bessell-B', Planck15)
    assert np.shape(mB) == (ng,)

    mB = uvista.apparent_magnitudes(coeff, z, multiple_filters, Planck15)
    assert np.shape(mB) == (ng, nf)

    # Test wrong number of coefficients
    nt_bad = 3
    coeff_bad = np.ones((ng, nt_bad))

    with pytest.raises(ValueError):
        MB = uvista.absolute_magnitudes(coeff_bad, 'bessell-B')

    with pytest.raises(ValueError):
        MB = uvista.absolute_magnitudes(coeff_bad, multiple_filters)

    with pytest.raises(ValueError):
        mB = uvista.apparent_magnitudes(coeff_bad, z, 'bessell-B', Planck15)

    with pytest.raises(ValueError):
        mB = uvista.apparent_magnitudes(coeff_bad, z, multiple_filters, Planck15)

    # Test stellar_mass parameter
    sm = [10, 20, 30, 40, 50, 60, 70]

    MB = uvista.absolute_magnitudes(coeff, 'bessell-B')
    MB_s = uvista.absolute_magnitudes(coeff, 'bessell-B', stellar_mass=sm)
    np.testing.assert_allclose(MB_s, MB - 2.5*np.log10(sm))

    MB = uvista.absolute_magnitudes(coeff, multiple_filters)
    MB_s = uvista.absolute_magnitudes(coeff, multiple_filters, stellar_mass=sm)
    np.testing.assert_allclose(MB_s, MB - 2.5*np.log10(sm)[:, np.newaxis])

    mB = uvista.apparent_magnitudes(coeff, z, 'bessell-B', Planck15)
    mB_s = uvista.apparent_magnitudes(coeff, z, 'bessell-B', Planck15, stellar_mass=sm)
    np.testing.assert_allclose(mB_s, mB - 2.5*np.log10(sm))

    mB = uvista.apparent_magnitudes(coeff, z, multiple_filters, Planck15)
    mB_s = uvista.apparent_magnitudes(coeff, z, multiple_filters, Planck15, stellar_mass=sm)
    np.testing.assert_allclose(mB_s, mB - 2.5*np.log10(sm)[:, np.newaxis])


def test_uvista_stellar_mass():
    import numpy as np
    from astropy import units
    from skypy_x.galaxies_paper import uvista
    from speclite.filters import FilterResponse

    # Gaussian bandpass
    filt_lam = np.logspace(3, 4, 1000) * units.AA
    filt_mean = 5000 * units.AA
    filt_width = 100 * units.AA
    filt_tx = np.exp(-((filt_lam-filt_mean)/filt_width)**2)
    filt_tx[[0, -1]] = 0
    FilterResponse(wavelength=filt_lam, response=filt_tx,
                   meta=dict(group_name='test', band_name='filt'))

    # Using the identity matrix for the coefficients yields trivial test cases
    coeff = np.eye(12)
    Mt = uvista.absolute_magnitudes(coeff, 'test-filt')

    # Using the absolute magnitudes of the templates as reference magnitudes
    # should return one solar mass for each template.
    stellar_mass = uvista.stellar_mass(coeff, Mt, 'test-filt')
    truth = 1
    np.testing.assert_allclose(stellar_mass, truth)

    # Solution for given magnitudes without template mixing
    Mb = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    stellar_mass = uvista.stellar_mass(coeff, Mb, 'test-filt')
    truth = np.power(10, -0.4*(Mb-Mt))
    np.testing.assert_allclose(stellar_mass, truth)
