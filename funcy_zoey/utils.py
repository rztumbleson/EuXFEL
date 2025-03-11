import h5py

def append_h5_data(filename, dataset_name, data, axis=0, chunksize=None):
    """
    Append data to an existing HDF5 file.

    Parameters:
    filename (str): The name of the HDF5 file.
    dataset_name (str): The name of the dataset to append to.
    data (numpy array): The data to append.
    axis (int): The axis along which to append the data.
    chunksize (tuple): The chunk size for the dataset.

    Returns:
    None
    """
    with h5py.File(filename, 'a') as f:
        # Check if the dataset exists
        if dataset_name in f:
            # Get the existing dataset
            dataset = f[dataset_name]
            # Append the new data to the existing dataset
            new_shape = list(dataset.shape)
            new_shape[axis] += data.shape[axis]
            dataset.resize(new_shape)
            slicing = [slice(None)] * len(new_shape)
            slicing[axis] = slice(-data.shape[axis], None)
            dataset[tuple(slicing)] = data
        else:
            # Create a new dataset if it doesn't exist
            maxshape = tuple(None if i == axis else s for i, s in enumerate(data.shape))
            f.create_dataset(dataset_name, data=data, maxshape=maxshape, chunks=chunksize)
        f.flush()
    return