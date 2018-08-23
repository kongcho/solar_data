from settings import *
from utils import *
from aperture import run_photometry, calculate_better_aperture
from plot import plot_data
from data import new_stars
from model import model

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def make_sound(duration=0.3, freq=440):
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    return 0

def main():
    logger.info("### starting ###")
    np.set_printoptions(linewidth=1000) #, precision=4)

    # target = run_photometry("7272437", plot_flag=False)
    # calculate_better_aperture(target, 0.001, 2, 0.7, 15)
    # plot_data(target)
    # plt.show()

    # kics = get_nth_kics("./data/table4.dat", 80001, 1, 0, " ", 0)
    # kics = get_nth_col("./data/table4.dat", 0, " ", 0)
    # n = new_stars(kics)

    # n.print_params("./res.out", ["variable", "curve_fit"])

    # print "VARIABLES LEN", len(n.variables)
    # print "NON VARIABLES LEN", len(n.non_variables)

    # n.plot_variable_params("luminosity", "teff")
    # plt.savefig("./lumvsteff.png")
    # plt.close("all")

    # n.plot_variable_bar("periodic")
    # plt.savefig("./periodic.png")
    # plt.close("all")

    # n.plot_variable_hist("prot")
    # plt.savefig("./prot.png")
    # plt.close("all")

    with open("./results/variable.out", "r") as f, open("./new.out", "w") as fout:
        r = csv.reader(f, delimiter=",")
        w = csv.writer(fout, delimiter=",")
        w.writerow(next(r) + ["ssr", "bic"])
        for row in r:
            kic = row[0]
            boolean = row[1]
            label = row[2]
            n = new_stars([kic])
            n._check_params(["lcs_new", "lcs_qs"])
            y, x, yerr = n._setup_lcs_xys(n.res[0])
            m = model(y, x, yerr=yerr)
            res = m.run_model(label)
            w.writerow([kic, boolean, label, res[0], res[1]])

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


