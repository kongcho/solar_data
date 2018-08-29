from settings import *
from utils import *
from aperture import run_photometry, calculate_better_aperture, model_background
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

    # target = run_photometry("12835232", plot_flag=False)
    # target.model_uncert()
    # calculate_better_aperture(target, 0.001, 2, 0.7, 15)
    # model_background(target, 0.2, 15)
    # print target.obs_flux
    # print target.target_uncert
    # plot_data(target)
    # plt.show()

    # kics = ["5042629", "4825845", "5854038", "5854073"]
    kics = ["2694810"
                , "4726114"
                , "7272437"
                , "11415049"
                , "10087863"
                , "3236788"
                , "8041424"
                , "8043142"
                , "8345997"
                , "8759594"
                , "10122937"
                , "3743810"
                , "4555566"
                , "5450764"
                , "6038355"
                , "6708110"
                , "7432092"
                , "7678238"
                , "8804069"
                , "9306271"
                , "11014223"
                , "11033434"
                , "11873617"
                , "12417799"
                , "5352687"
                , "6263983"
                , "7433192"
    ]


    # kics = get_nth_col("./data/table4.dat", 0, " ", 0)
    # kics = get_nth_kics("./data/table4.dat", 1000, 1, 0, " ", 0)

    # n = new_stars(kics)

    # n.plot_variable_params("luminosity", "teff")
    # plt.savefig("./lumvsteff.png")
    # plt.close("all")

    # param = "luminosity"
    # n.plot_variable_hist(param)
    # plt.savefig("./%s.png" % param)
    # plt.close("all")

    # for param in ["periodic"]:
    #     n.plot_variable_bar(param)
    #     plt.savefig("./%s.png" % param)
    #     plt.close("all")

    # for param in ["dist", "logg", "mass", "metallicity", "prot", "rad", "teff", "rho", \
    #               "av", "rper"]:
    #     n.plot_variable_hist(param)
    #     plt.savefig("./%s.png" % param)
    #     plt.close("all")

    # with open("./results/lc_data_var.out", "r") as f:
    #     r = csv.reader(f, delimiter=",")
    #     next(r)
    #     for row in r:
    #         kic = row[0]
    #         boolean = row[1]
    #         label = row[2]
    #         n = new_stars([kic])
    #         n._check_params(["lcs_new", "lcs_qs"])
    #         y, x, yerr = n._setup_lcs_xys(n.res[0])
    #         m = model(y, x, yerr=yerr)
    #         res = m.run_model(label)
    #         print row, res


    kics = ["4726114", "10087863", "6263983"]
    for kic in kics:
        n = new_stars([kic])
        n.get_params(["lcs_new", "lcs_qs"])

        pars = n.res[0]["params"]
        y, x, yerr = n._setup_lcs_xys(n.res[0])
        m = model(y, x, yerr=yerr, qs=pars["qs"])
        m.is_variable()
        m.plot_many_lines()
        plt.show()

    # n = new_stars(kics)
    # n.get_is_variable()
    # for i, star in enumerate(n.res):
    #     pars = star["params"]
        # print star["kic"], pars["variable"], pars["curve_fit"]


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


