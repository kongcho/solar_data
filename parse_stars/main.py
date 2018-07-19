from model import *
from utils import *
from settings import *
from aperture import *
from data import *
from parse import *
from plot import *

logger = setup_main_logging()

import os
import numpy as np

def make_sound(duration=0.3, freq=440):
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    return 0

def main():
    logger.info("### starting ###")
    # np.set_printoptions(linewidth=1000, precision=4)

    kics = (get_nth_kics(filename_stellar_params, 4010, 1, ' ', 0))[:]
    # kics = get_kics(filename_stellar_params, " ", 0)

    print_lc_improved(kics[:10], "lc_data.out")

    make_sound(0.3, 440)
    logger.info("### everything done ###")
    return 0

if __name__ == "__main__":
    os.sys.path.append(f3_location)
    from f3 import photometry
    main()
