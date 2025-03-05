import sys
sys.path.append('../')
from funcy_zoey import photon_maps as zphotons


if __name__ == '__main__':
    zphotons.generate_photon_maps(run=70, module=0, overwrite_file=False)