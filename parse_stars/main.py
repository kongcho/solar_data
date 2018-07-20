from settings import filename_stellar_params, f3_location
from utils import get_kics, make_sound
from plot import print_lc_improved

import os
import numpy as np


def main():
    logger.info("### starting ###")
    # np.set_printoptions(linewidth=1000, precision=4)

    # kics = (get_nth_kics(filename_stellar_params, 4010, 1, ' ', 0))[:]
    kics = get_kics(filename_stellar_params, " ", 0)[:]

    # print_lc_improved(kics, ("/data/ffidata/results/lc_data_img.out" \
    #                          , "/data/ffidata/results/lc_data_old.out" \
    #                          , "/data/ffidata/results/lc_data_new.out"))

    make_sound(0.3, 440)
    logger.info("### everything done ###")
    return 0

if __name_ == "__main__":
    os.sys.path.append(f3_location)
    from f3 import photometry

    from settings import setup_main_logging
    logger = setup_main_logging()
    main()

