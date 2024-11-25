from scipy import special as sp
from lmfit import Model, Parameters
import numpy as np

def neg_binom(k_bar, k, M):
    """
    Calculate the negative binomial distribution for photon-arrival events. Details can
    be found in Speckle Phenomena in Optics: Theory and Applications by Goodman. See section
    on speckle at low light levels.

    Parameters:
    k_bar (float): The average number of photons.
    k (int): The number of photons.
    M (int): The number of modes influencing the measurement.

    Returns:
    float: The probability of k photons arriving given an average incident intensity
    (k_bar) and number of modes (M) .
    """
    # Calculate the prefactor using the log-gamma function for numerical stability
    prefactor = np.exp(sp.loggamma(k + M) - sp.loggamma(M) - sp.loggamma(k+1))
    postfactor = ((k_bar/(k_bar+M))**k)*((M/(k_bar+M))**M) # no this is not a factor....

    return prefactor * postfactor
    
    
def fit_contrasts(data, k_bar, min_k=0, max_k=None):
    """
    Fit a negative binomial model to the data to estimate the contrast parameter M.

    Parameters:
    data (numpy array): The data to be fit.
    k_bar (numpy array): The mean number of photons.
    min_k (int, optional): The minimum number of photons to include in the fit. Defaults to 0.
    max_k (int, optional): The maximum number of photons to include in the fit. Defaults to None.

    Returns:
    tuple: The best-fit value of M and its standard error.
    """

    # Remove P=0 (no photon) events
    mask = data > 0

    # Create a broadcasted array of k values
    k_fit = np.broadcast_to(np.arange(data.shape[-1]), data.shape)

    # Apply the min_k and max_k filters to the mask
    mask[k_fit < min_k] = False
    if max_k is not None:
        mask[k_fit > max_k] = False

    # Flatten the mask
    mask = np.ravel(mask)

    # Flatten the data, k_bar, and k_fit arrays and apply the mask
    Y = np.ravel(data)[mask]
    X = np.ravel(k_bar)[mask]
    k = np.ravel(k_fit)[mask]

    # Define the negative binomial model
    model = Model(neg_binom, independent_vars=['k_bar', 'k'])

    # Define the model parameters
    params = Parameters()
    params.add('M', value=4, min=0.001, max=10000)

    # Fit the model to the data
    result = model.fit(data=Y, k_bar=X, k=k, **params, nan_policy='omit')

    # Return the best-fit value of M and its standard error
    return result.best_values['M'], result.params['M'].stderr


def modes_to_contrast(M, M_err):
    """
    Convert the number of modes (M) to contrast and propagate the error in contrast.

    Parameters:
    M (float): The number of modes.
    M_err (float): The error in the number of modes.

    Returns:
    tuple: The contrast and its error.
    """

    # Calculate the contrast from the number of modes
    # The contrast is inversely proportional to the square root of the number of modes
    contrast = 1 / np.sqrt(M)

    # Calculate the error in contrast using the error propagation formula
    # The error in contrast is proportional to the contrast itself and the relative error in the number of modes
    contrast_err = 0.5 * contrast * M_err / M

    # Return the contrast and its error
    return contrast, contrast_err


def load_labels(module, q_cutoff=15, return_unique_labels=True, return_counts=False):
    """
    Load labels from an HDF5 file and apply a q-cutoff.

    Parameters:
    module (int): The module number.
    q_cutoff (int, optional): The q-cutoff value. Defaults to 15.
    return_unique_labels (bool, optional): Whether to return unique labels. Defaults to True.
    return_counts (bool, optional): Whether to return the counts of each unique label. Defaults to False.

    Returns:
    tuple: The labels, unique labels, and counts (if requested).
    """

    # Open the HDF5 file containing the labels
    ds = xr.open_dataset('./labels.h5')

    # Extract the q-grid values for the specified module
    q = np.squeeze(ds['q_grid'][ds.module_order==module].to_numpy())

    # Extract the labels for the specified module
    labels = np.squeeze(ds['labels'][ds.module_order==module].to_numpy())

    # Apply the q-cutoff to the labels
    # Set labels with q-values above the cutoff to NaN
    labels[q>q_cutoff] = np.nan

    # If requested, return the unique labels
    if return_unique_labels:
        # If counts are also requested, return the unique labels and their counts
        if return_counts:
            # Get the unique labels and their counts
            unique_labels, counts = np.unique(labels, return_counts=return_counts)
            # Remove NaN values from the unique labels and counts
            counts = counts[~np.isnan(unique_labels)] 
            unique_labels = unique_labels[~np.isnan(unique_labels)] 
            # Return the labels, unique labels, and counts
            return labels, unique_labels, counts
        else:
            # Get the unique labels
            unique_labels = np.unique(labels)
            # Remove NaN values from the unique labels
            unique_labels = unique_labels[~np.isnan(unique_labels)] 
            # Return the labels and unique labels
            return labels, unique_labels
    else:
        # Return only the labels
        return labels