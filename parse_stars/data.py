"""
FUNCTIONS THAT GETS RIGHT PARAMETERS FOR EACH STAR / FILTER PARAMETERS
"""

from utils import get_nth_kics
from settings import setup_logging
from api import api
from aperture import run_photometry, calculate_better_aperture, model_background
from model import model
logger = setup_logging()

from math import pi
import matplotlib.pyplot as plt

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
        return 0

    def _calc_luminosity(self, radius, teff):
        sb_const = float("5.670367e-08")
        return sb_const*4*pi*(radius**2)*(teff**4)

    def get_luminosity(self):
        if "teff" not in self.params:
            self.get_params(["teff"])
        if "rad" not in self.params:
            self.get_params(["rad"])
        for i, star in enumerate(self.res):
            star_pars = star["params"]
            star_pars["luminosity"] = self._calc_luminosity(star_pars["rad"], star_pars["teff"])
        return 0

    def get_basic_params(self, neighbour_arcsep=0.15):
        base_params = ["teff", "logg", "metallicity", "rad", "mass" "rho", "dist", "av", \
                       "prot", "rper", "periodic", "neighbours", "close_edges"]
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

    def filter_params(self, param_dic):
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
        return res

    def _build_errorbars(self, x, err, qs):
        error_list = np.zeros_like(x)
        for i in range(len(err)):
            g = np.where(qs == i)[0]
            error_list[g] += err[i]
        return 0

    def _setup_xys(self, star):
        if "lcs_new" not in self.params:
            self.get_params(["lcs_new"])
        if "lcs_qs" not in self.params:
            self.get_params(["lcs_qs"])

        star_pars = star["params"]
        y_dat = star_pars["lcs"]
        x_dat = star_pars["times"]
        yerr = star_pars["target_uncert"]
        # yerr = self._build_errorbars(x, star_pars["flux_uncert"], star_pars["qs"])

        good_idxs = [i for i in range(len(y_dat)) if y_dat[i] is not None]
        y = [y_dat[i] - 1 for i in good_idxs]
        x = [x_dat[i] for i in good_idxs]
        return y, x, yerr

    def get_is_variable(self):
        for i, star in enumerate(self.res):
            print "ADUHAOSDSAOID", star["kic"]
            y, x, yerr = self._setup_xys(star)
            m = model(y, x, yerr=yerr)
            star["params"]["variable"] = m.is_variable()
            m.plot_many_lines()
            plt.show()
            plt.close("all")
        return 0

    def plot_variable_params(self, param1, param2):
        for i, star in enumerate(self.res):
            pass

    def plot_variable_freq(self, param):
        pass

if __name__ == "__main__":
    kics = get_nth_kics("./data/table4.dat", 10000, 1, 0, " ", 0)

    n = new_stars(kics)
    n.get_is_variable()
