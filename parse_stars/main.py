from settings import *
from utils import *
from aperture import run_photometry
from data import new_stars
import os
import numpy as np
import csv

def make_sound(duration=0.3, freq=440):
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    return 0



def main():
    logger.info("### starting ###")
    # np.set_printoptions(linewidth=1000, precision=4)

    # base_params = ["teff", "logg", "metallicity", "rad", "mass" "rho", "dist", "av", \
    #                "periodic"]
    # param_dic = {"neighbours": [lambda x: x != []]}
    # kics = ["757280", "757450"]
    # n = new_stars(kics)

    # run_photometry("893233")

    make_sound(0.8, 440)
    logger.info("### everything done ###")
    return 0

if __name__ == "__main__":
    os.sys.path.append(f3_location)
    from f3 import photometry

    from settings import setup_main_logging
    logger = setup_main_logging()
    main()


