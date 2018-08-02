import settings
from utils import get_kics
from parse import table_api
from mast import mast_api
logger = settings.setup_logging()

import kplr

class api(object):

    def __init__(self):
        self.sources = ["q1-17", "mast", "mast_table", "periodic", "nonperiodic", \
                        "lc_img", "lc_new", "lc_old"]
        self.updated_dir = settings.filename_stellar_params
        self.nonperiodic_dir = settings.filename_nonperiods
        self.periodic_dir = settings.filename_periods
        self.mast_table_dir = settings.filename_mast_table
        self.lc_img_dir = settings.filename_lc_img
        self.lc_new_dir = settings.filename_lc_new
        self.lc_old_dir = settings.filename_lc_old

    def get_params(self, kics, params, **neighbour_filters):
        new_params = [{} for _ in kics]
        updated_params = ["teff", "logg", "metallicity", "rad", "mass" "rho", "dist", "av"]
        updated_pars, periodic_pars, mast_pars, mast_table_pars = ([] for _ in range(4))

        for param in params:
            if param in updated_params:
                updated_pars.append(param)
            elif param in settings.updated_dic.keys():
                updated_pars.append(param)
            else:
                mast_pars.append(param)

        if "periodic" in params:
            self._update_params(new_params, self.get_periodic_or_not(kics))
        if "neighbours" in params:
            self._update_params(new_params, self.get_neighbours_or_not(kics, **neighbour_filters))
        if "lcs_new" in params:
            self._update_params(new_params, self.get_lcs_times_uncerts(kics, self.lc_new_dir))
        if "lcs_old" in params:
            self._update_params(new_params, self.get_lcs_times_uncerts(kics, self.lc_old_dir))
        if "lcs_img" in params:
            self._update_params(new_params, self.get_lcs_imgs(kics))
        self._update_params(new_params, self.get_updated_params(kics, updated_pars))
        self._update_params(new_params, self.get_mast_params(kics, mast_pars))
        return new_params

    def _update_params(self, old_res, new_res):
        for i, res in enumerate(new_res):
            old_res[i].update(res)
        return old_res

    def _format_params(self, fields_dic, params):
        fields = []
        types = []
        all_keys = fields_dic.keys()
        for param in params:
            if param in all_keys:
                call_name, field_type = fields_dic[param]
                fields.append(call_name)
                types.append(field_type)
        return fields, types

    def get_updated_params(self, kics, params):
        col_name_arr, type_arr = self._format_params(settings.updated_dic, params)
        t = table_api(self.updated_dir, " ", 0, 0)
        col_nos = t.get_col_nos(params, col_name_arr)
        param_res = t.parse_table_arrs(col_nos, kics, type_arr)
        return param_res

    def get_periodic_or_not(self, kics):
        periodic = get_kics(self.periodic_dir, ",", 1)
        unperiodics = get_kics(self.nonperiodic_dir, ",", 1)
        reses = []
        for kic in kics:
            curr_params = {}
            if kic in periodic:
                curr_params["periodic"] = True
            elif kic in unperiodics:
                curr_params["periodic"] = False
            else:
                curr_params["periodic"] = "Unsure"
            reses.append(curr_params)
        return reses

    def get_periodic_params(self, kics, params):
        col_name_arr, type_arr = self._format_params(settings.periodic_dic, params)
        t = table_api(self.updated_dir, ",", 1, 0)
        col_nos = t.get_col_nos(params, col_name_arr)
        param_res = t.parse_table_arrs(col_nos, kics, type_arr)
        return param_res

    def get_nonperiodic_params(self, kics, params):
        col_name_arr, type_arr = self._format_params(settings.nonperiodic_dic, params)
        t = table_api(self.updated_dir, ",", 1, 0)
        col_nos = t.get_col_nos(params, col_name_arr)
        param_res = t.parse_table_arrs(col_nos, kics, type_arr)
        return param_res

    def get_mast_params(self, kics, params):
        client = kplr.API()
        reses = []
        for kic in kics:
            curr_params = {}
            star = client.star(kic)
            target = client.target(kic)
            for param in params:
                if param in star.params.keys():
                    curr_params[param] = star.params[param]
                elif param in target.params.keys():
                    curr_params[param] = target.params[param]
            reses.append(curr_params)
        return reses

    def get_neighbours_or_not(self, kics, **filter_params):
        reses = []
        for kic in kics:
            curr_params = {}
            neighbours = []
            m = mast_api()
            res = m.parse_json_output("kepler", "kic10", ["Kepler ID"], target=kic, **filter_params)
            for i in res:
                neighbours.append(i["Kepler ID"])
            curr_params["neighbours"] = neighbours
            reses.append(curr_params)
        return reses

    def get_lcs_times_uncerts(self, kics, lc_file):
        t = table_api(lc_file, delimiter=",", skip_rows=1, kic_col_no=0)
        arrs = t.parse_table_arrs(range(1, 109), kics=kics, types=[float]*108)
        reses = []
        for i, res in enumerate(kics):
            curr_params = {}
            arr = arrs[i]
            if len(arr) == 0:
                curr_params["lcs"], curr_params["flux_uncert"], \
                    curr_params["target_uncert"], curr_params["times"] = (None for _ in range(4))
                logger.error("couldn't parse table for this kic: %s" % kic)
            else:
                curr_params["lcs"] = arr[:52]
                curr_params["flux_uncert"] = arr[52:56]
                curr_params["target_uncert"] = arr[56:108]
                curr_params["times"] = map(float, t.get_nth_row(1)[1:53])
            reses.append(curr_params)
        return reses

    def get_lcs_imgs(self, kics):
        t = table_api(lc_file, delimiter=",", skip_rows=1, kic_col_no=0)
        arrs = t.parse_table_arrs(range(1, 902), kics=kics, types=[float]*902)
        reses = []
        for i, res in enumerate(kics):
            curr_params = {}
            arr = arrs[i]
            if len(arr) == 0:
                curr_params["aperture"] = None
                logger.error("couldn't parse table for this kic: %s" % kic)
            else:
                curr_params["aperture"] = arr
            reses.append(curr_params)
        return reses

