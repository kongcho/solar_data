from settings import setup_logging
from api import api
from aperture import run_photometry, calculate_better_aperture, model_background
logger = setup_logging()

class new_stars(object):

    def __init__(self, kics):
        table_file = "./tests/lc_data_new.out"

        self.kics = kics
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
        a = api()
        params_arr = a.get_params(self.kics, params, **kwargs)
        for i, star in enumerate(self.new_star):
            star["params"].update(params_arr[i])
        return 0

    def get_basic_params(self, neighbour_arcsep=0.15):
        base_params = ["teff", "logg", "metallicity", "rad", "mass" "rho", "dist", "av", \
                       "periodic", "neighbours"]
        self.get_params(base_params, radius=neighbour_arcsep)
        return 0

    def get_neighbour_res(self):
        completed = self.kics
        for i, star in enumerate(self.res):
            pass
        for neighbour in neighbours:
            if neighbour not in self.kics:
        pass

    def filter_params(self, kics, param_dic):
        res_kics = []
        res_kics_idxs = []
        res_kics_bools = [True] * len(kics)
        for i, kic in enumerate(self.kics):
            if kic in kics:
                res_kics_idxs.append(i)
                res_kics.append(kic)
        for i, kic in enumerate(res_kics):
            kic_params = self.params[i]
            for param in param_dic:
                for boolf in param_dic[param]:
                    if res_kics_bools[i] and not boolf(kic_params[param]):
                        res_kics_bools[i] = False
        res = [res_kics[i] for i in range(len(res_kics)) if res_kics_idxs[i]]
        return res

