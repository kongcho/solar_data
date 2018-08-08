from settings import *
from utils import *
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

    # kics = ["757280", "757450"]
    # n = new_stars(kics)
    # print n.get_basic_params(0.15)
    # print n.get_other_params("lc_new")
    # print n.res

    all_kics = get_kics("./kics.out")
    calc_kics = get_kics("/data/results/lc_data_new.out")
    non_kics = list(set(all_kics) ^ set(calc_kics))
    simple_array_to_file("./failed_kics.txt", non_kics)

    make_sound(0.8, 440)
    logger.info("### everything done ###")
    return 0

if __name__ == "__main__":
    os.sys.path.append(f3_location)
    from f3 import photometry

    from settings import setup_main_logging
    logger = setup_main_logging()
    main()


