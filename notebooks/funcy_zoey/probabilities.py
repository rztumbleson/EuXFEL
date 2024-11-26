import numba as nb
import numpy as np

@nb.jit(nopython=True, parallel=True)
def bincount(arr, max_photons=19):
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

@nb.jit(nopython=True, parallel=True)
def calculate_probabilities_chunk(chunk, train_sum, intensity_threshold=0.05, pulses_per_train=200, max_photons=19):
    """
    Iterate over delays and calculate the bincount for each delay.

    Parameters:
    chunk (numpy array): Input chunk with shape (n, pulses_per_train, ...).
    pulses_per_train (int, optional): Number of pulses per train. Defaults to 200.
    max_photons (int, optional): Maximum number of photons. Defaults to 19.

    Returns:
    numpy array: Output array with shape (pulses_per_train, ..., max_photons+1).
    """
    
    out = np.zeros((pulses_per_train, chunk.shape[2], chunk.shape[3], max_photons+1), dtype=np.float32)
    for dt in nb.prange(pulses_per_train):
        if dt == 0:
            filt = np.full((train_sum.shape[0], pulses_per_train), True)
            filt[:, :pulses_per_train-1] = get_intensity_filter(train_sum, 1, thresh=intensity_threshold)
        else:
            filt = get_intensity_filter(train_sum, dt, thresh=intensity_threshold)
        two_sum = add_two_stacks(chunk, dt, mask=filt)
        out[dt] = bincount(two_sum, max_photons=max_photons) / filt.sum()
    return out

@nb.jit(nopython=True, parallel=True, cache=True)
def get_intensity_filter(arr, dt, thresh, pulses_per_train=200):
    out = np.full((arr.shape[0], pulses_per_train - dt), True)
    for i in range(arr.shape[0]):
        for j in range(pulses_per_train - dt):
            left = arr[i, j]
            right = arr[i, j + dt]
            out[i, j] = np.abs(left - right) / ((left + right) / 2) < thresh
    return out

def load_train_sum():
    # todo: actually save properly
    return np.load('../data/train_sum_masked.npy')

