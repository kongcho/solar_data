from settings import *
from utils import *
from aperture import *
from plot import *
from data import new_stars

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def make_sound(duration=0.3, freq=440):
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    return 0


def photo(kic):
    target = run_photometry(kic, plot_flag=True)
    target.model_uncert()
    # calculate_better_aperture(target, 0.001, 2, 0.7, 15)
    # model_background(target, 0.2, 15)
    print target.obs_flux
    print target.target_uncert
    plot_data(target)
    plt.show()

def plot(kics):
    n = new_stars(kics)

    n.plot_variable_params("luminosity", "teff")
    plt.savefig("./lumvsteff.png")
    plt.close("all")

    param = "luminosity"
    n.plot_variable_hist(param)
    plt.savefig("./%s.png" % param)
    plt.close("all")

    for param in ["periodic"]:
        n.plot_variable_bar(param)
        plt.savefig("./%s.png" % param)
        plt.close("all")

    for param in ["dist", "logg", "mass", "metallicity", "prot", "rad", "teff", "rho", \
                  "av", "rper"]:
        n.plot_variable_hist(param)
        plt.savefig("./%s.png" % param)
        plt.close("all")

def model_var(kics):
    with open("./results/lc_data_var.out", "r") as f:
        r = csv.reader(f, delimiter=",")
        next(r)
        for row in r:
            kic = row[0]
            boolean = row[1]
            label = row[2]
            n = new_stars([kic])
            n._check_params(["lcs_new", "lcs_qs"])
            y, x, yerr = n._setup_lcs_xys(n.res[0])
            m = model(y, x, yerr=yerr)
            res = m.run_model(label)
            print row, res

def print_models(kics):
    for kic in kics:
        n = new_stars([kic])
        n.get_params(["lcs_new", "lcs_qs"])

        pars = n.res[0]["params"]
        y, x, yerr = n._setup_lcs_xys(n.res[0])
        m = model(y, x, yerr=yerr, qs=pars["qs"])
        m.is_variable()
        m.plot_many_lines()
        plt.show()

def output(kics, n):
    params = ["variable", "curve_fit", "var_chi2", "var_bic", "var_res"]
    fout = "var_out_%d.out" % n
    ns = new_stars(kics)
    ns.print_params(fout, params)
    print "-------------------- OUTPUT %d DONE" % n
    return 0

def randfor(kics, factor):
    features = ["teff", "dist", "rad", "avs", "evState", "binaryFlag", \
                "logg", "metallicity", "rho", "av", \
                "prot", "rper", "LPH", "closest_edge"]

    n = new_stars(kics)
    res = n.do_random_forest(features, factor)
    if res != 1:
        params = n.prune_params(RandomForestClassifier, res[0], res[1])
        print "PARAMSHERE", params
        res_new = n.do_random_forest(params, factor)

    return 0

def plot_hists(n, tot, params):
    kics = get_nth_col("./data/table4.dat", 0, " ", 0)
    n = new_stars(kics)

    n.plot_variable_params("luminosity", "teff")
    plt.savefig("./lumvsteff.png")
    plt.close("all")

    param = "luminosity"
    n.plot_variable_hist(param)
    plt.savefig("./%s.png" % param)
    plt.close("all")

    for param in ["periodic"]:
        n.plot_variable_bar(param)
        plt.savefig("./%s.png" % param)
        plt.close("all")

    for param in ["dist", "logg", "mass", "metallicity", "prot", "rad", "teff", "rho", \
                  "av", "rper"]:
        n.plot_variable_hist(param)
        plt.savefig("./%s.png" % param)
        plt.close("all")




def main():
    logger.info("### starting ###")
    np.set_printoptions(linewidth=1000) #, precision=4)

    kics = get_nth_col("./data/table4.dat", 0, " ", 0)[:1]
    # kics = get_nth_kics("./data/table4.dat", 5000, 1, 0, " ", 0)
    # kics = ["9083355"]

    # features = ["teff", "dist", "rad", "avs", "evState", "binaryFlag", \
    #             "logg", "metallicity", "rho", "av", \
    #             "prot", "rper", "LPH", "closest_edge"]

    # randfor(1)

    # n = new_stars(kics)
    # n.get_is_variable()

    # kics = ["757099", "893730", "893750","1026669","1027030","1028012","1028640",\
    #         "1160947", "1161345","1161432","1163211"]
    # kics = ["893234", "1026647", "1163579"]

    do_multiprocess(1, run_kics, output, kics)
    # randfor(kics, 1)
    # randfor(kics, 0.7)

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


