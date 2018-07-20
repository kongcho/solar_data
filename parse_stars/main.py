from settings import filename_stellar_params, f3_location
from utils import get_kics
from plot import print_lc_improved

import os
import numpy as np


def make_sound(duration=0.3, freq=440):
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    return 0


def main():
    logger.info("### starting ###")
    # np.set_printoptions(linewidth=1000, precision=4)

    # kics = (get_nth_kics(filename_stellar_params, 4010, 1, ' ', 0))[:]
    kics = get_kics(filename_stellar_params, " ", 0)[:]

    # print_lc_improved(kics, ("/data/ffidata/results/lc_data_img.out" \
    #                          , "/data/ffidata/results/lc_data_old.out" \
    #                          , "/data/ffidata/results/lc_data_new.out"))

    print_lc_improved(kics, ("./tests/lc_data_img.out" \
                             , "./tests/lc_data_old.out" \
                             , "./tests/lc_data_new.out"))

    make_sound(0.3, 440)
    logger.info("### everything done ###")
    return 0

if __name__ == "__main__":
    os.sys.path.append(f3_location)
    from f3 import photometry

    from settings import setup_main_logging
    logger = setup_main_logging()
    main()

