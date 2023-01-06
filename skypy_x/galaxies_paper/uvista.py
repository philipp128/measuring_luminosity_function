r"""Galaxy spectrum module.

"""

from astropy import units
import numpy as np
from skypy.utils.photometry import SpectrumTemplates
from pathlib import Path


class UltraVistaTemplates(SpectrumTemplates):
    num_templates = 12
    template_file = 'tweak_fsps_temp_kc13_12_{:03d}.dat.gz'
    lam_unit = units.Unit('angstrom')
    flam_unit = units.Unit('erg s-1 cm-2 angstrom-1')
    # TODO: check this is the correct V-band
    MLv_band = 'bessell-V'

    def __init__(self):
        data = Path(__file__).parent / 'data' / 'uvista_nmf'
        self.wavelength = np.loadtxt(data / 'lambda.def') * self.lam_unit
        self.templates = np.zeros((self.num_templates, self.wavelength.size)) * self.flam_unit
        for i in range(self.num_templates):
            temp_lam, temp_flam = np.loadtxt(data / self.template_file.format(i+1)).T
            assert np.allclose(temp_lam, self.wavelength.value)
            self.templates[i] = temp_flam * self.flam_unit
        param = np.genfromtxt(data / 'param.def', names=True)
        Lv = 10**(0.4*self.absolute_magnitudes(np.eye(self.num_templates), self.MLv_band))
        self.templates /= (Lv*param['MLv'])[:, np.newaxis]

    def stellar_mass(self, coefficients, magnitudes, filter):
        r'''Compute stellar mass from absolute magnitudes in a reference filter.

        This function takes composite spectra for a set of galaxies defined by
        template fluxes *per unit stellar mass* and multiplicative coefficients
        and calculates the stellar mass required to match given absolute
        magnitudes for a given bandpass filter in the rest frame.

        Parameters
        ----------
        coefficients : (ng, nt) array_like
            Array of template coefficients.
        magnitudes : (ng,) array_like
            The magnitudes to match in the reference bandpass.
        filter : str
            A single reference bandpass filter specification for
            `~speclite.filters.load_filters`.

        Returns
        -------
        stellar_mass : (ng,) array_like
            Stellar mass of each galaxy in template units.
        '''
        Mt = self.absolute_magnitudes(coefficients, filter)
        return np.power(10, 0.4*(Mt-magnitudes))


uvista = UltraVistaTemplates()
