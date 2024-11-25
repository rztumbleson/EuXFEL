import numpy as np
import dask.array as da
import xarray as xr

def convert_to_real_photon_maps(run, module, group, gain=0.2):
    """
    During the Lingjia/Zoey meeting on 10/18/2024 we noticed that the low q 
    photon intensity was too. After further investigating, it appears that 
    the photon maps saved in the folder: scratch/LS/Reduced_Data/reduced_3-17-23/
    requires an additional gain multiplication and rounding to be converted to 
    actual photon arrival events. All data processed before 10/18/2024 using 
    the photon statistics method should not be trusted.
    
    Convert the "*_photon.npy" data that was saved in the reduced_3-17-23 folder
    to (hopefully) real photon maps with discrete values.

    Parameters:
    run (int): The run number of the experiment.
    module (int): The module number of the detector.
    group (int): The group number of the data.
    gain (float, optional): The gain factor to apply to the photon data. Defaults to 0.2.

    Returns:
    discrete_photon (np.ndarray): The discrete photon map with values in the range [0, 255].

    Notes:
    The function loads photon data from a numpy file, applies a gain factor, and then 
    discretizes the values to the range [0, 255].
    """
    folder = f'/gpfs/exfel/u/scratch/SCS/202201/p002884/LS/Reduced_Data/reduced_3-17-23/'
    photon = np.load(folder + f'r{run:0d}m{module:0d}_{group:1d}_photon.npy', 'r')
    photon = np.minimum(photon, 255)
    discrete_photon = np.rint(gain*photon).astype(np.uint8)
    return discrete_photon

def load_photon_maps_dask(run, module, group, train_index):
    photon_maps_filename = f'/gpfs/exfel/u/scratch/SCS/202201/p002884/LS/Reduced_Data/reduced_3-17-23/r{run:0d}m{module:0d}_{group:1d}_photon.npy'
    photon_maps = np.load(photon_maps_filename, 'r')[train_index]
    return da.from_array(photon_maps)


def load_labels(module, q_cutoff=15, return_unique_labels=True, return_counts=False):
    ds = xr.open_dataset('./labels.h5')
    q = np.squeeze(ds['q_grid'][ds.module_order==module].to_numpy())
    labels = np.squeeze(ds['labels'][ds.module_order==module].to_numpy())
    labels[q>q_cutoff] = np.nan
    
    if return_unique_labels:
        if return_counts:
            unique_labels, counts = np.unique(labels, return_counts=return_counts)
            counts = counts[~np.isnan(unique_labels)] 
            unique_labels = unique_labels[~np.isnan(unique_labels)] 
            return labels, unique_labels, counts
        else:
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[~np.isnan(unique_labels)] 
            return labels, unique_labels
    else:
        return labels
    
    