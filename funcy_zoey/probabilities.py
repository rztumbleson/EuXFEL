import numba as nb
import numpy as np
import h5py
from tqdm import tqdm

@nb.jit(nopython=True, parallel=True)
def bincount(arr, max_photons):
    """
    Apply np.bincount to each pixel in the chunk.

    Parameters:
    arr (numpy array): Input chunk with shape (n, x, y).
    max_photons (int, optional): Maximum number of photons. Defaults to 19.

    Returns:
    numpy array: Output array with shape (x, y, max_photons+1).
    """
    if arr.ndim != 3:
        raise ValueError('Array dimension not 3')

    n_sample, n_i, n_j = arr.shape
    out = np.zeros((n_i, n_j, max_photons+1), dtype=np.uint64)
    for i in nb.prange(n_i):
        for j in nb.prange(n_j):
            for x in nb.prange(n_sample):
                if arr[x, i, j] <= max_photons:
                    out[i, j, arr[x, i, j]] += 1
    return out


@nb.jit(nopython=True, parallel=False)
def add_two_stacks(arr, delay, mask=None, pulses_per_train=200):
    """
    Add two stacks of pulses with a given delay.

    Parameters:
    arr (numpy array): Input array with shape (n, pulses_per_train, ...).
    delay (int): Delay between the two stacks.
    mask (numpy array, optional): Mask to apply to the input array. Defaults to None.
    pulses_per_train (int, optional): Number of pulses per train. Defaults to 200.

    Returns:
    numpy array: Output array with shape (n, pulses_per_train - delay, ...).
    """
    n_i = len(arr)
    n_j = pulses_per_train - delay
    
    if arr.shape[1] != pulses_per_train:
        raise ValueError('pulses_per_train not axis 1')
    
    if mask is None:
        mask = np.full((n_i, n_j), True)
    n = np.sum(mask)
    out = np.empty((n, *arr.shape[2:]), dtype=arr.dtype)
    idx = 0
    for i in nb.prange(n_i):
        for j in nb.prange(n_j):
            if mask[i, j]:
                out[idx] = arr[i, j] + arr[i, j + delay]     
                idx += 1
    return out


@nb.jit(nopython=True, parallel=True, cache=True)
def get_relative_intensity_filter(arr, dt, thresh, pulses_per_train=200):
    out = np.full((arr.shape[0], pulses_per_train - dt), True)
    for i in nb.prange(arr.shape[0]):
        for j in nb.prange(pulses_per_train - dt):
            left = arr[i, j]
            right = arr[i, j + dt]
            out[i, j] = np.abs(left - right) / ((left + right) / 2) < thresh
    return out

def load_train_sum(run, module):
    # todo: actually save properly
    return np.load(f'../data/small_data/train_sum_run{run:03d}_module{module:02d}.npy')

def load_photon_maps(run, module, filter=None):
    input_path = f'../data/small_data/run{run:03d}_module{module:02d}_photon_maps.h5'
    if filter is None:
        filter = np.s_[:]
    with h5py.File(input_path, 'r') as f:
        photon_maps = f['photon_maps'][filter]
    return photon_maps

def filter_full_trains():
    filt = np.full((100), False)
    filt[np.random.choice(100, 30)] = True
    return filt

def generate_pmfs(run, module):
    arr = load_train_sum(run, module)[:100]
    filt_abs = filter_full_trains()
    photon_maps = load_photon_maps(run, module, filter=filt_abs)
    max_photons = np.amax(photon_maps) * 2

    pmfs = np.zeros((200, 128, 512, max_photons+1), dtype=np.float32)
    for dt in tqdm(range(200)):
        filt_rel = get_relative_intensity_filter(arr[filt_abs], dt, thresh=0.05)
        two_sum = add_two_stacks(photon_maps, dt, mask=filt_rel)
        pmfs[dt] = bincount(two_sum, max_photons=max_photons) / filt_rel.sum()
    return pmfs