import os
import csv
import itertools
import numpy as np

import matplotlib
if os.environ.get('DISPLAY','') == "":
    print("Using non-interactive Agg backend")
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from matplotlib.backends.backend_pdf import PdfPages

from astropy.modeling import models, fitting

from model import *
from utils import *
from settings import *
from aperture import *
from data import *
from parse import *
from plot import *

logger = setup_main_logging()

def print_lc(kics, fout, image_region=15):
    is_first = True
    with open(fout, "w") as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        for kic in kics:
            target = run_photometry(kic)
            if target == 1:
                continue
            if is_first:
                names = ["KIC"] + build_arr_n_names("img", 900) + \
                        map(str, target.times) + build_arr_n_names("uncert_old", 4)
                writer.writerow(names)
                is_first = False
            calculate_better_aperture(target, 0.001, 2, 0.7, image_region=image_region)
            arr = np.concatenate([np.asarray([kic]), target.img.flatten(), \
                                  target.obs_flux, target.flux_uncert])
            model_background(target, 0.2, 15)
            # arr = np.append(arr, target.obs_flux)
            # arr = np.append(arr, target.flux_uncert)
            writer.writerow(arr)
            logger.info("done: %s" % kic)
    logger.info("done")
    return 0


def testing(targ):
    target = run_photometry(targ)
    return target.times, target.obs_flux

def make_sound(duration=0.3, freq=440):
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    return 0

def main():
    logger.info("### starting ###")
    ## gets good kics to be put into MAST
    # array_to_file(get_good_existing_kids(filename_stellar_params, get_kics(filename_periods)))

    ## uses MAST results to get stars without neighbours
    # remove_bright_neighbours_separate()

    ## plots list of files with kics using f3
    # plot_targets(targets_file, [is_large_ap, has_peaks], get_kics(targets_file, ' ', 0))


    ## TEMP
    ben_kics = ["2694810"
                , "3236788"
                , "3743810"
                , "4555566"
                , "4726114"
                , "5352687"
                , "5450764"
                , "6038355"
                , "6263983"
                , "6708110"
                , "7272437"
                , "7432092"
                , "7433192"
                , "7678238"
                , "8041424"
                , "8043142"
                , "8345997"
                , "8759594"
                , "8804069"
                , "9306271"
                , "10087863"
                , "10122937"
                , "11014223"
                , "11033434"
                , "11415049"
                , "11873617"
                , "12417799"
                ]

    ## TESTS
    np.set_printoptions(linewidth=1000) #, precision=1)

    # kics = (get_nth_kics(filename_stellar_params, 4000, 1, ' ', 0))[:]
    kics = ["11913365"] #, "11913377"] + ben_kics

    # SIMPLE TESTS
    # for kic in kics:
    #     print testing(kic)
    print_lc(kics, "out_pc.csv")

    make_sound(0.3, 440)
    logger.info("### everything done ###")
    return 0

if __name__ == "__main__":
    os.sys.path.append(f3_location)
    from f3 import photometry
    main()
