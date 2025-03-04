from glob import glob
import re
import h5py
import os
from tqdm.notebook import tqdm
from datetime import datetime
import numpy as np
from extra_data import open_run # type: ignore


def initialize_h5_photon_maps(output_file, run, module):
    """
    Create an HDF5 file with SWMR mode enabled for storing
    photon maps and initialize its metadata.

    Parameters:
    file_list (list): List of files containing photon data.
    chunk_size (tuple): Chunk size for the HDF5 dataset.
    run (int): Run number of the experiment.
    module (int): Module number of the detector.
    output_file (str): Output file path to save the HDF5 dataset.
    """

    # Experimental parameters for saving metadata
    current = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    dark_runNB = [69,  72,  72,  75,  75,  78,  78,  81,  81]
    runNB =      [70,  71,  73,  74,  76,  77,  79,  80,  82]
    index = runNB.index(run)

    # Create the HDF5 file and initialize its metadata
    with h5py.File(output_file, 'w', libver='latest') as f:
        f.swmr_mode = True
        # Set the metadata attributes
        f.attrs['run'] = run
        f.attrs['dark_run'] = dark_runNB[index]
        f.attrs['module'] = module
        f.attrs['proposal'] = 2884
        f.attrs['field'] = current[index]
        f.attrs['date_processed'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    return


def generate_photon_maps(run, module, overwrite_file=False):
    """
    During the Lingjia/Zoey meeting on 10/18/2024 we noticed that the low q 
    photon intensity was too high. After further investigating, it appears that 
    the photon maps saved in the folder: scratch/LS/Reduced_Data/reduced_3-17-23/
    is not actual discrete photon data. All data processed before 11/04/2024 using 
    the photon statistics method should not be trusted.
    
    Generates photon maps with discrete values in HDF5 format.

    Parameters:
    run (int): Run number of the experiment.
    module (int): Module number of the detector.
    overwrite_file (bool, optional): Whether to overwrite the existing file. Defaults to False.
    """
    # Define the output file path
    output_file = f'../data/photon_maps/run{run:03d}_module{module:02d}.h5'

    # Check if the output file already exists
    if not overwrite_file and os.path.exists(output_file):
        print('Output file already exists.')
        return
    else:
        # Create the HDF5 file and initialize its metadata
        initialize_h5_photon_maps(output_file, run, module)

    
    dark_run = get_dark_run_number(run)    
    pedastal_file = f'../data/pedastals/run{dark_run:03d}_module{module:02d}.npy'
    pedastal = get_pedastal(pedastal_file, dark_run=dark_run, module=module, pulses_per_train=400)
    distribute_photon_map_computation(run, module, pedastal, pulses_per_train=400, output_file=output_file)
    return


def get_dark_run_number(run):
    """
    Get the dark run number corresponding to the given run number.

    Parameters:
    run (int): Run number of the experiment.

    Returns:
    int: Dark run number.
    """
    dark_runNB = [69,  72,  72,  75,  75,  78,  78,  81,  81]
    runNB =      [70,  71,  73,  74,  76,  77,  79,  80,  82]
    index = runNB.index(run)
    return dark_runNB[index]

def distribute_photon_map_computation(run, module, pedastal, pulses_per_train, output_file):
    """
    Distribute the photon map computation across multiple chunks. This does not take advantage
    of any parallelization currently.

    Parameters:
    run (int): Run number of the experiment.
    module (int): Module number of the detector.
    pedastal (numpy array): Pedestal values.
    pulses_per_train (int): Number of pulses per train.
    output_file (str): Output file path to save the HDF5 dataset.
    """
    dask_trains = get_trains(run, module, pulses_per_train)
    for chunk_start in tqdm(range(0, dask_trains.shape[0], 100), desc='Generating Photon Maps (chunks)'):
        chunk_stop = min(chunk_start+100, dask_trains.shape[0])
        chunk = np.s_[chunk_start:chunk_stop]
        arr = process_chunk(dask_trains[chunk], pedastal)
        append_h5_data(output_file, 'photon_maps', arr.compute())
    return
    
def process_chunk(arr, pedastal, gain=0.199142159721309):
    """
    Process a chunk of trains data to generate photon maps.

    Parameters:
    arr (dask array): Chunk of trains data.
    pedastal (numpy array): Pedestal values.
    gain (float, optional): Gain value. Defaults to 0.199142159721309.

    Returns:
    dask array: Processed photon maps.
    """
    arr = arr.astype(float)
    arr = arr - np.expand_dims(pedastal, axis=(0))
    arr = arr[:,::2] - arr[:,1::2]
    arr = arr * gain
    arr = np.clip(arr, 0, 255)
    arr = arr.astype(np.uint8)
    return arr

def get_trains(run, module, pulses_per_train, proposal=2884):
    """
    Get the dask array of trains for the given run and module.

    Parameters:
    run (int): Run number of the experiment.
    module (int): Module number of the detector.
    pulses_per_train (int): Number of pulses per train.
    proposal (int, optional): Proposal number. Defaults to 2884.

    Returns:
    dask array: Trains data.
    """
    run_ed = open_run(proposal=proposal, run=run)
    arr = run_ed.get_dask_array(f'SCS_DET_DSSC1M-1/DET/{module}CH0:xtdf', 'image.data')[:, 0]
    arr = arr.reshape(-1, pulses_per_train, 128, 512)
    return arr


def get_pedastal(pedastal_file, dark_run=None, module=None, pulses_per_train=None, proposal=2884):
    """
    Get the pedestal values for the given dark run and module. If this has been computed already
    and is stored in pedastal_file then the file is loaded, otherwise calculates the pedastal and
    and saves it in pedastal_file.

    Parameters:
    pedastal_file (str): File path to save the pedestal values.
    dark_run (int, optional): Dark run number. Defaults to None.
    module (int, optional): Module number. Defaults to None.
    pulses_per_train (int, optional): Number of pulses per train. Defaults to None.
    proposal (int, optional): Proposal number. Defaults to 2884.

    Returns:
    numpy array: Pedestal values.
    """
    if os.path.exists(pedastal_file):
        pedastal = np.load(pedastal_file)
    else:
        if dark_run == None or pulses_per_train == None:
            print('Provide pedastal run and number of pulses.')
            return
        dark_run_ed = open_run(proposal=proposal, run=dark_run)
        arr_dark = dark_run_ed.get_dask_array(f'SCS_DET_DSSC1M-1/DET/{module}CH0:xtdf', 'image.data')[:, 0]
        arr_dark = arr_dark.reshape(-1, pulses_per_train, 128, 512)
        pedastal = arr_dark.astype(float).mean(axis=0).compute()
        np.save(pedastal_file, pedastal)
    return pedastal
    


def append_h5_data(filename, dataset_name, data):
    """
    Append data to an existing HDF5 file.

    Parameters:
    filename (str): The name of the HDF5 file.
    dataset_name (str): The name of the dataset to append to.
    data (numpy array): The data to append.

    Returns:
    None
    """
    with h5py.File(filename, 'a', libver='latest') as f:
        # Check if the dataset exists
        if dataset_name in f:
            # Get the existing dataset
            dataset = f[dataset_name]
            # Append the new data to the existing dataset
            dataset.resize((dataset.shape[0] + data.shape[0], dataset.shape[1], dataset.shape[2], dataset.shape[3]))
            dataset[-len(data):, :, :, :] = data
        else:
            # Create a new dataset if it doesn't exist
            f.create_dataset(dataset_name, data=data, maxshape=(None, data.shape[1], data.shape[2], data.shape[3]))
        f.flush()
    return