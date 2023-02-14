import importlib
import sys
import time
import scipy
print(time.ctime() + '\n')

print(sys.version + '\n')

for module in ['numpy', 'scipy', 'matplotlib','astropy','eazy', 'prospect']:
    #print(module)
    mod = importlib.import_module(module)
    print('{0:>20} : {1}'.format(module, mod.__version__))
    
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
# read number of tasks
n_tasks = int(sys.argv[1])
print(f"Number of tasks: {n_tasks}")
import eazy

# Symlink templates & filters from the eazy-code repository
try:
    print('EAZYCODE = '+os.getenv('EAZYCODE'))
except:
    pass

if not os.path.exists('templates'):
    eazy.symlink_eazy_inputs() 
    
# quiet numpy/astropy warnings
import warnings
from astropy.utils.exceptions import AstropyWarning

np.seterr(all='ignore')
warnings.simplefilter('ignore', category=AstropyWarning)

# on lustre
parameter_file = 'parameters_sciama.txt'
translate_file = 'translate.txt'

self = eazy.photoz.PhotoZ(param_file=parameter_file, translate_file=translate_file, zeropoint_file=None, 
                          load_prior=True, load_products=False)

sn = self.fnu/self.efnu
clip = (sn > 1).sum(axis=1)
print(sn)
print(clip)
print(self.idx[clip])

print(self.fnu.shape)
print(self.efnu)

print(self.RES[314])
print(self.flux_columns)
print(self.err_columns)
print(self.f_numbers)
print(self.NFILT)
print(self.cat['HSC_g_FLUX_APER3'])
print(self.ok_data)

NITER = 3
NBIN = np.minimum(self.NOBJ//100, 180)
print('NBIN: ', NBIN)
print('NOBJ: ', self.NOBJ)
print(self.NOBJ//100)

self.param.params['VERBOSITY'] = 1.
for iter in range(NITER):
    print('Iteration: ', iter)
    
    sn = self.fnu/self.efnu
    clip = (sn > 1).sum(axis=1) > 4 # Generally make this higher to ensure reasonable fits
    self.iterate_zp_templates(idx=self.idx[clip], update_templates=False, 
                              update_zeropoints=True, iter=iter, n_proc=n_tasks, 
                              save_templates=False, error_residuals=False, 
                              NBIN=NBIN, get_spatial_offset=False)
    
# Turn off error corrections derived above
self.set_sys_err(positive=True)

# Full catalog
sample = np.isfinite(self.ZSPEC)

# fit_parallel renamed to fit_catalog 14 May 2021
self.fit_catalog(self.idx[sample], n_proc=n_tasks)

# Derived parameters (z params, RF colors, masses, SFR, etc.)
warnings.simplefilter('ignore', category=RuntimeWarning)
zout, hdu = self.standard_output(simple=False, 
                                 rf_pad_width=0.5, rf_max_err=2, 
                                 prior=True, beta_prior=True, 
                                 absmag_filters=[], 
                                 extra_rf_filters=[])

# 'zout' also saved to [MAIN_OUTPUT_FILE].zout.fits

zout.info()

print(zout['z_spec'])
print(zout['z_phot'])