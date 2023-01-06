from astropy.table import Table
import numpy as np


def cosmos_coefficients(redshift, classification, file, bins=None):
    cosmos_table = Table.read(file, cache=True)[1].data
    mask_cosmos_class = cosmos_table['CLASS'] == classification
    bins = [0, np.max(cosmos_table['z_eval'])] if bins is None else bins
    coefficients = np.zeros(len(redshift), np.shape(cosmos_table['COEFFS'])[1])
    for z_min, z_max in zip(bins[:-1], bins[1:]):
        mask_cosmos_z = np.logical_and(z_min <= cosmos_table['z_eval'], cosmos_table['z_eval'] < z_max)
        cosmos_coefficients_z = cosmos_table['COEFFS'][np.logical_and(mask_cosmos_class, mask_cosmos_z)]
        n_cosmos_z = len(cosmos_coefficients_z)
        mask_sample_z = np.logical_and(z_min <= redshift, redshift < z_max)
        n_sample_z = np.count_nonzero(mask_sample_z)
        choice = np.random.choice(n_cosmos_z, size=n_sample_z, replace=True)
        coefficients[mask_sample_z] = cosmos_coefficients_z[choice]
    return coefficients
