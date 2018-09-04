"""
FUNCTIONS TO INTERACT WITH DATABASES TO GET PARAMETERS FOR EACH STAR
"""

import settings
from utils import get_nth_col
from parse import table_api
from mast import mast_api
logger = settings.setup_logging()

import kplr
import numpy as np

class api(object):

    def __init__(self):
        self.sources = ["q1-17", "mast", "mast_table", "periodic", "nonperiodic", \
                        "lc_img", "lc_new", "lc_old"]
        self.updated_dir = settings.filename_stellar_params
        self.nonperiodic_dir = settings.filename_nonperiods
        self.periodic_dir = settings.filename_periods
        self.mast_table_dir = settings.filename_mast_table
        self.gaia_dir = settings.filename_gaia_table
        self.lc_img_dir = settings.filename_lc_img
        self.lc_new_dir = settings.filename_lc_new
        self.lc_old_dir = settings.filename_lc_old
        self.lc_obs_dir = settings.filename_lc_obs
        self.lc_var_dir = settings.filename_lc_var

    def get_params(self, kics, params, **neighbour_filters):
        new_params = [{} for _ in kics]
        gaia_pars, updated_pars, periodic_pars, mast_pars, mast_table_pars = ([] for _ in range(5))

        for param in params:
            if param in settings.gaia_dic.keys():
                gaia_pars.append(param)
            elif param in settings.updated_dic.keys():
                updated_pars.append(param)
            elif param in settings.periodic_dic.keys():
                periodic_pars.append(param)
            elif param in settings.mast_params:
                mast_pars.append(param)

        if "periodic" in params:
            self._update_params(new_params, self.get_periodic_or_not(kics))
        if "neighbors" in params:
            self._update_params(new_params, self.get_neighbours_or_not(kics, **neighbour_filters))
        if "lcs_new" in params:
            self._update_params(new_params, self.get_lcs_times_uncerts(kics, self.lc_new_dir))
        if "lcs_old" in params:
            self._update_params(new_params, self.get_lcs_times_uncerts(kics, self.lc_old_dir))
        if "lcs_img" in params:
            self._update_params(new_params, self.get_lcs_imgs(kics, self.lc_img_dir))
        if "lcs_qs" in params:
            self._update_params(new_params, self.get_lcs_qs(kics, self.lc_obs_dir))
        if "close_edges" in params:
            self._update_params(new_params, self.get_close_edges(kics))
        if "variable" in params:
            self._update_params(new_params, self.get_variable(kics, self.lc_var_dir))

        try:
            self._update_params(new_params, self.get_mast_params(kics, mast_pars))
        except Exception as e:
            logger.error("can't retrieve mast params: %s-%s" % (e.message, mast_pars))
        # self._update_params(new_params, self.get_nonperiodic_params(kics, periodic_pars))
        self._update_params(new_params, self.get_periodic_params(kics, periodic_pars))
        self._check_params_dic(new_params, periodic_pars)
        self._update_params(new_params, self.get_updated_params(kics, updated_pars))
        self._update_params(new_params, self.get_gaia_params(kics, gaia_pars))
        self._check_params_dic(new_params, gaia_pars)
        return new_params

    def _update_params(self, old_res, new_res):
        for i, res in enumerate(new_res):
            old_res[i].update(res)
        return old_res

    def _format_params(self, fields_dic, params):
        fields = []
        types = []
        for param in params:
            if param in fields_dic.keys():
                call_name, field_type = fields_dic[param]
                fields.append(call_name)
                types.append(field_type)
        return fields, types

    def _check_params_dic(self, reses, params):
        for i, res in enumerate(reses):
            for param in params:
                if param not in res.keys():
                    res[param] = np.nan
        return reses

    def get_gaia_params(self, kics, params):
        col_name_arr, type_arr = self._format_params(settings.gaia_dic, params)
        t = table_api(self.gaia_dir, "&", 1, 0, "\\")
        col_nos = t.get_col_nos(col_name_arr, settings.gaia_dic_keys)
        param_res = t.parse_table_dicts(col_nos, kics, type_arr, params)
        param_res = self._check_params_dic(param_res, params)
        return param_res

    def get_updated_params(self, kics, params):
        col_name_arr, type_arr = self._format_params(settings.updated_dic, params)
        t = table_api(self.updated_dir, " ", 0, 0)
        col_nos = t.get_col_nos(col_name_arr, settings.updated_dic_keys)
        param_res = t.parse_table_dicts(col_nos, kics, type_arr, params)
        param_res = self._check_params_dic(param_res, params)
        return param_res

    def get_periodic_params(self, kics, params):
        col_name_arr, type_arr = self._format_params(settings.periodic_dic, params)
        t = table_api(self.periodic_dir, ",", 1, 0)
        col_nos = t.get_col_nos(col_name_arr, settings.periodic_dic_keys)
        param_res = t.parse_table_dicts(col_nos, kics, type_arr, params)
        return param_res

    def get_nonperiodic_params(self, kics, params):
        col_name_arr, type_arr = self._format_params(settings.nonperiodic_dic, params)
        t = table_api(self.nonperiodic_dir, ",", 1, 0)
        col_nos = t.get_col_nos(col_name_arr, settings.nonperiodic_dic_keys)
        param_res = t.parse_table_dicts(col_nos, kics, type_arr, params)
        return param_res

    def get_periodic_or_not(self, kics):
        periodic = get_nth_col(self.periodic_dir, 0, ",", 1)
        unperiodics = get_nth_col(self.nonperiodic_dir, 0, ",", 1)
        reses = []
        for kic in kics:
            curr_params = {}
            if kic in periodic:
                curr_params["periodic"] = 1 # True
            elif kic in unperiodics:
                curr_params["periodic"] = 0 # False
            else:
                curr_params["periodic"] = 2 # Unsure
            reses.append(curr_params)
        return reses

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
                if i["Kepler ID"] != kic:
                    neighbours.append(i["Kepler ID"])
            curr_params["neighbors"] = neighbours
            reses.append(curr_params)
        return reses

    def get_close_edges(self, kics, min_distance=20, shape=(1070, 1132)):
        reses = []
        rows_ks = ["Row_0", "Row_1", "Row_2", "Row_3"]
        cols_ks = ["Column_0", "Column_1", "Column_2", "Column_3"]
        mast_reses = self.get_mast_params(kics, keys)
        for res in mast_reses:
            curr_params = {}
            edges = []
            rows = [res[k] if res[k] is not None else np.nan for k in rows_ks]
            cols = [res[k] if res[k] is not None else np.nan for k in cols_ks]
            if np.nanmin(rows) <= min_distance:
                edges.append("Top")
            elif abs(shape[0]-np.nanmax(rows)) <= min_distance:
                edges.append("Bottom")
            if np.nanmin(cols) <= min_distance:
                edges.append("Left")
            elif abs(shape[0]-np.nanmax(rows)) <= min_distance:
                edges.append("Right")
            if not edges:
                edges.append(None)
            curr_params["close_edges"] = edges
            reses.append(curr_params)
        return reses

    def get_lcs_times_uncerts(self, kics, lc_file):
        t = table_api(lc_file, delimiter=",", skip_rows=1, kic_col_no=0)
        arrs = t.parse_table_arrs(range(1, 109), kics=kics, types=[float]*108)
        reses = []
        for i, kic in enumerate(kics):
            curr_params = {}
            arr = arrs[i]
            if len(arr) == 0:
                curr_params["lcs"], curr_params["flux_uncert"], \
                    curr_params["target_uncert"], curr_params["times"] = (np.nan for _ in range(4))
                logger.error("couldn't parse table for this kic: %s" % kic)
            else:
                curr_params["lcs"] = arr[:52]
                curr_params["flux_uncert"] = arr[52:56]
                curr_params["target_uncert"] = arr[56:108]
                curr_params["times"] = map(float, t.get_nth_row(1)[1:53])
            reses.append(curr_params)
        return reses

    def get_lcs_imgs(self, kics, lc_file):
        t = table_api(lc_file, delimiter=",", skip_rows=1, kic_col_no=0)
        arrs = t.parse_table_arrs(range(1, 902), kics=kics, types=[float]*902)
        reses = []
        for i, kic in enumerate(kics):
            curr_params = {}
            arr = arrs[i]
            if len(arr) == 0:
                curr_params["aperture"] = None
                logger.error("couldn't parse table for this kic: %s" % kic)
            else:
                curr_params["aperture"] = arr
            reses.append(curr_params)
        return reses

    def get_lcs_qs(self, kics, lc_file):
        t = table_api(lc_file, delimiter=" ", skip_rows=0, kic_col_no=None)
        times = t.get_nth_col(0, float)
        qs = t.get_nth_col(1, int)
        years = t.get_nth_col(2, int)
        curr_params = {"qs": qs, "years": years} # same pointer, unchanged data
        reses = [curr_params]*len(kics)
        return reses

    def get_variable(self, kics, lc_file):
        t = table_api(lc_file, delimiter=",", skip_rows=1, kic_col_no=0)
        arrs = t.parse_table_arrs(range(1, 3), kics=kics, types=[str, str, float, float])
        reses = []
        for i, kic in enumerate(kics):
            curr_params = {}
            arr = arrs[i]
            if len(arr) == 0:
                curr_params["variable"], curr_params["curve_fit"], \
                    curr_params["var_chi2"], curr_params["var_bic"] = (np.nan for _ in range(4))
                logger.error("couldn't parse table for this kic: %s" % kic)
            else:
                curr_params["variable"] = 1 if arr[0] == "True" else 0
                curr_params["curve_fit"] = arr[1]
                # curr_params["var_chi2"] = arr[2]
                # curr_params["var_bic"] = arr[3]
            reses.append(curr_params)
        return reses

if __name__ == "__main__":
    a = api()
    kics = ["757280", "757450"]
    b = a.get_mast_params(kics, ["g_KIS"])
    print b
