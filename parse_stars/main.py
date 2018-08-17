from settings import *
from utils import *
from aperture import run_photometry, calculate_better_aperture
from data import new_stars
import os
import numpy as np
import csv
from plot import plot_data, print_lc_improved
import matplotlib.pyplot as plt
import itertools
import kplr

def make_sound(duration=0.3, freq=440):
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    return 0

def main():
    logger.info("### starting ###")
    np.set_printoptions(linewidth=1000) #, precision=4)

    # replace_lines_fix("./tests/old.txt", "./tests/new.txt", "./tests/dat.txt")

    # base_params = ["teff", "logg", "metallicity", "rad", "mass" "rho", "dist", "av", \
    #                "periodic"]
    # param_dic = {"neighbours": [lambda x: x != []]}
    # kics = ["757280", "757450"]
    # n = new_stars(kics)
    # n.get_basic_params(0.15)
    # print n.res

    target = run_photometry("757137", plot_flag=False)
    calculate_better_aperture(target, 0.001, 2, 0.7, 15)
    plot_data(target)
    plt.show()

    make_sound(0.8, 440)
    logger.info("### everything done ###")
    return 0

if __name__ == "__main__":
    os.sys.path.append(f3_location)
    from f3 import photometry

    from settings import setup_main_logging
    logger = setup_main_logging()

    mpl_setup()
    main()


