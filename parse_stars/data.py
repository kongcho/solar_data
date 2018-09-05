"""
FUNCTIONS THAT GETS RIGHT PARAMETERS FOR EACH STAR / FILTER PARAMETERS
"""

from utils import get_nth_kics
from settings import setup_logging, mpl_setup
from api import api
from aperture import run_photometry, calculate_better_aperture, model_background
from model import model
logger = setup_logging()

from math import pi, log
from collections import Counter
import csv
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_selection import RFECV
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

class new_stars(object):

    def __init__(self, kics):
        self.kics = kics
        self.params = []
        self.res = [{} for _ in kics]
        for i, star in enumerate(self.res):
            star["kic"] = self.kics[i]
            star["params"] = {}

    def make_targets(self):
        for star in self.res:
            target = run_photometry(self.kics)
            if target != 1:
                target.model_uncert()
                calculate_better_aperture(target, 0.001, 2, 0.7, 15)
                model_background(target, 0.2, 15)
            star["target"] = target
        return 0

    def get_params(self, params, **kwargs):
        self.params += params
        a = api()
        params_arr = a.get_params(self.kics, params, **kwargs)
        for i, star in enumerate(self.res):
            star["params"].update(params_arr[i])
        logger.info("done: %s", params)
        return 0

    def _check_params(self, params):
        rem_pars = []
        for param in params:
            if param not in self.params:
                if param == "luminosity":
                    self.get_luminosity()
                elif param == "variable":
                    # self.get_is_variable()

                    self.get_params([param])
                    self.setup_self_variable()
                elif param == "closest_edge":
                    self.get_edge_distance()
                else:
                    rem_pars.append(param)
        self.get_params(rem_pars)
        return 0

    def _calc_luminosity(self, radius, teff):
        sb_const = float("5.670367e-08")
        solar_lum_const = float("7.942e8")
        return log(sb_const*4*pi*(radius**2)*(teff**4)/solar_lum_const, 10)

    def get_luminosity(self):
        self._check_params(["teff", "rad"])
        for i, star in enumerate(self.res):
            star_pars = star["params"]
            star_pars["luminosity"] = self._calc_luminosity(star_pars["rad"], star_pars["teff"])
        self.params.append("luminosity")
        return 0

    def get_basic_params(self, neighbour_arcsep=0.15):
        base_params = ["teff", "logg", "metallicity", "rad", "mass" "rho", "dist", "av", \
                       "prot", "rper", "periodic", "neighbors", "close_edges"]
        self.get_params(base_params, radius=neighbour_arcsep)
        return 0

    def get_neighbour_res(self, params=[], **kwargs):
        for i, star in enumerate(self.res):
            star_pars = star["params"]
            star_pars["neighbours_stars"] = []
            if star_pars["neighbours"]:
                for neighbour in star_pars["neighbours"]:
                    new_star = new_stars([neighbour])
                    new_star.get_params(params, **kwargs)
                    star_pars["neighbours_stars"].append(new_star)
        return 0

    def get_edge_distance(self):
        self._check_params(["close_edges"])
        for i, star in enumerate(self.res):
            star_pars = star["params"]
            edges = star_pars["close_edges"]
            star_pars["closest_edge"] = np.nanmin(edges)
        self.params.append("closest_edge")
        return 0

    def filter_params(self, param_dic, edit_res=True):
        res_kics = []
        res_kics_idxs = []
        res_kics_bools = [True] * len(self.kics)
        for i, kic in enumerate(self.kics):
            if kic in kics:
                res_kics_idxs.append(i)
                res_kics.append(kic)
        for i, kic in enumerate(res_kics):
            kic_params = self.res[i]["params"]
            for param in param_dic:
                for boolf in param_dic[param]:
                    if res_kics_bools[i] and not boolf(kic_params[param]):
                        res_kics_bools[i] = False
        res = [self.res[res_kics_idxs[i]] for i in range(len(res_kics)) if res_kics_bools[i]]
        if edit_res:
            self.res = res
        return res

    def _setup_lcs_xys(self, star):
        star_pars = star["params"]
        y_dat = star_pars["lcs"]
        x_dat = star_pars["times"]
        yerr = star_pars["target_uncert"]
        # yerr = self._build_errorbars(x, star_pars["flux_uncert"], star_pars["qs"])

        good_idxs = [i for i in range(len(y_dat)) if y_dat[i] is not None]
        y = [y_dat[i] - 1 for i in good_idxs]
        x = [x_dat[i] for i in good_idxs]
        return y, x, yerr

    def _build_errorbars(self, x, err, qs):
        error_list = np.zeros_like(x)
        for i in range(len(err)):
            g = np.where(qs == i)[0]
            error_list[g] += err[i]
        return 0

    def get_is_variable(self):
        self.variables = []
        self.non_variables = []
        self._check_params(["lcs_new", "lcs_qs"])
        for i, star in enumerate(self.res):
            y, x, yerr = self._setup_lcs_xys(star)
            m = model(y, x, yerr=yerr, qs=star["params"]["qs"])
            res, label, bic, ssr = m.is_variable()
            star["params"]["variable"] = res
            star["params"]["curve_fit"] = label
            star["params"]["var_bic"] = bic
            star["params"]["var_chi2"] = ssr
            if res:
                self.variables.append(star["kic"])
            else:
                self.non_variables.append(star["kic"])
            logger.info("done: %s, %s" % (star["kic"], label))
        self.params += ["variable", "curve_fit", "var_bic", "var_chi2"]
        return 0

    def setup_self_variable(self):
        self._check_params(["variable"])
        self.variables = []
        self.non_variables = []
        for i, star in enumerate(self.res):
            if star["params"]["variable"]:
                self.variables.append(star["kic"])
            else:
                self.non_variables.append(star["kic"])
        return 0

    def plot_variable_params(self, paramy, paramx):
        self._check_params(["variable", paramy, paramx])
        non_var_xs = []
        non_var_ys = []
        var_xs = []
        var_ys = []
        for i, star in enumerate(self.res):
            if star["kic"] in self.variables:
                var_ys.append(star["params"][paramy])
                var_xs.append(star["params"][paramx])
            non_var_ys.append(star["params"][paramy])
            non_var_xs.append(star["params"][paramx])

            # elif star["kic"] in self.non_variables:
            #     non_var_ys.append(star["params"][paramy])
            #     non_var_xs.append(star["params"][paramx])

        plt.plot(non_var_xs, non_var_ys, "bx", label="non_variable")
        plt.plot(var_xs, var_ys, "rx", label="variable")
        plt.legend(loc="upper right")
        plt.ylabel(paramy)
        plt.xlabel(paramx)
        plt.title("%s vs %s" % (paramy, paramx))

        logger.info("done")
        return 0

    def plot_variable_bar(self, param):
        self._check_params(["variable", param])
        non_var_xs = []
        var_xs = []
        for i, star in enumerate(self.res):
            if star["kic"] in self.variables:
                var_xs.append(star["params"][param])
            # elif star["kic"] in self.non_variables:
            #     non_var_xs.append(star["params"][param])
            non_var_xs.append(star["params"][param])

        non_var_dic = Counter(non_var_xs)
        var_dic = Counter(var_xs)
        all_keys = list(set(non_var_dic.keys()+var_dic.keys()))
        var_keys_hs = range(len(all_keys))
        non_var_keys_hs = [x+0.2 for x in var_keys_hs]
        var_xs_hs = [var_dic[k] if k in var_dic.keys() else 0 for k in all_keys]
        non_var_xs_hs = [non_var_dic[k] if k in non_var_dic.keys() else 0 for k in all_keys]

        plt.bar(var_keys_hs, var_xs_hs, width=0.2, color="r", alpha=0.5, \
                label="variable", tick_label=all_keys)
        plt.bar(non_var_keys_hs, non_var_xs_hs, width=0.2, color="b", alpha=0.5, \
                label="non_variable")
        plt.legend(loc="upper right")
        plt.xlabel(param)
        plt.ylabel("frequency")
        plt.title("%s" % (param))

        logger.info("done")
        return 0

    def plot_variable_hist(self, param):
        self._check_params(["variable", param])
        non_var_xs = []
        var_xs = []
        for i, star in enumerate(self.res):
            if star["kic"] in self.variables:
                var_xs.append(star["params"][param])
            # elif star["kic"] in self.non_variables:
            #     non_var_xs.append(star["params"][param])
            non_var_xs.append(star["params"][param])

        var_xs = np.array(var_xs)
        non_var_xs = np.array(non_var_xs)
        var_xs = var_xs[~np.isnan(var_xs)]
        non_var_xs = non_var_xs[~np.isnan(non_var_xs)]

        if not np.any(var_xs) and not np.any(non_var_xs):
            logger.error("no data points for this param")
            return 1

        plt.hist(var_xs, color=["r"], alpha=0.5, label="variable")
        plt.hist(non_var_xs, color=["b"], alpha=0.5, label="non variable")
        plt.legend(loc="upper right")
        plt.xlabel(param)
        plt.ylabel("frequency")
        plt.title("%s" % (param))

        logger.info("done")
        return 0

    def print_params(self, fout, params):
        self._check_params(params)
        header = ["kic"] + params
        with open(fout, "w") as f:
            w = csv.writer(f, delimiter=",", lineterminator="\n")
            w.writerow(header)
            for i, star in enumerate(self.res):
                arr = [star["kic"]]
                for param in params:
                    arr.append(star["params"][param])
                w.writerow(arr)
        logger.info("done")
        return 0

    def _setup_skl(self, params):
        self._check_params(["variable"] + params)
        variables = []
        data = []
        for i, star in enumerate(self.res):
            pars = star["params"]
            good_pars = []
            for par in params:
                if par not in pars.keys():
                    break
                if np.isnan(pars[par]):
                    break
                else:
                    good_pars.append(pars[par])
            else:
                data.append(good_pars)
                variables.append(pars["variable"])
        return data, variables

    def prune_params(self, fitting, data, variables):
        rfe = RFECV(fitting, verbose=1)
        rfe.fit(data, variables)
        param_rank = sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), params))
        logger.info("done: %s" % param_rank)
        return params[rfe.support_]

    def do_random_forest(self, params, train_factor):
        clf = RandomForestClassifier()
        all_dat, all_vars = self._setup_skl(params)
        if not all_dat and not all_vars:
            logger.error("not enough data")
            return 1

        if train_factor == 1:
            train_x, train_y = all_dat, all_vars
            test_x, test_y = all_dat, all_vars
        else:
            train_x, test_x, \
                train_y, test_y = train_test_split(all_dat, all_vars, train_size=train_factor)

        clf = clf.fit(train_x, train_y)
        param_rank = sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), params))
        logger.info("done: %s" % param_rank)
        logger.info("accuracy (all data): %f" % clf.score(all_dat, all_vars))
        logger.info("accuracy (test data): %f" % clf.score(test_x, test_y))
        return train_x, train_y


if __name__ == "__main__":
    mpl_setup()

    kics = get_nth_kics("./data/table4.dat", 80001, 1, 0, " ", 0)

    n = new_stars(kics)
    n._setup_logreg([])
