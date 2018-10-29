import settings
from utils import get_nth_col
from parse import table_api
from mast import mast_api
logger = settings.setup_logging()

import kplr
import numpy as np
import ast

class api(object):
    """
    api to get necessary parameters from combined databases
    """

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
        """
        gets given parameters for list of kics into list of dictionaries
        see possible parameters in settings.py

        :kics: list of kics to get parameters for
        :params: list of parameters to get for each kic
        :neighbour_filters: parameter filters to determine what a neighbour is

        :return: list of dictionary of parameters
        """
        new_params = [{} for _ in kics]
        gaia_pars, updated_pars, periodic_pars, mast_pars, mast_table_pars = ([] for _ in range(5))

        for param in params:
            if param in settings.gaia_dic.keys():
                gaia_pars.append(param)
            if param in settings.updated_dic.keys():
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
        if "close_labels" in params:
            self._update_params(new_params, self.get_close_labels(kics, 20))
        elif "close_edges" in params:
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
        """
        gets new parameters from Gaia mission from [1]
        data preceeds all other data from other databases
        """
        col_name_arr, type_arr = self._format_params(settings.gaia_dic, params)
        t = table_api(self.gaia_dir, "&", 1, 0, "\\")
        col_nos = t.get_col_nos(col_name_arr, settings.gaia_dic_keys)
        param_res = t.parse_table_dicts(col_nos, kics, type_arr, params)
        return param_res

    def get_updated_params(self, kics, params):
        """
        gets new parameters from Mathur table [2]
        """
        col_name_arr, type_arr = self._format_params(settings.updated_dic, params)
        t = table_api(self.updated_dir, " ", 0, 0)
        col_nos = t.get_col_nos(col_name_arr, settings.updated_dic_keys)
        param_res = t.parse_table_dicts(col_nos, kics, type_arr, params)
        param_res = self._check_params_dic(param_res, params)
        return param_res

    def get_periodic_params(self, kics, params):
        """
        gets parameters for periodic stars from McQuillian [3]
        """
        col_name_arr, type_arr = self._format_params(settings.periodic_dic, params)
        t = table_api(self.periodic_dir, ",", 1, 0)
        col_nos = t.get_col_nos(col_name_arr, settings.periodic_dic_keys)
        param_res = t.parse_table_dicts(col_nos, kics, type_arr, params)
        return param_res

    def get_nonperiodic_params(self, kics, params):
        """
        gets parameters for non-periodic stars from McQuillian [3]
        """
        col_name_arr, type_arr = self._format_params(settings.nonperiodic_dic, params)
        t = table_api(self.nonperiodic_dir, ",", 1, 0)
        col_nos = t.get_col_nos(col_name_arr, settings.nonperiodic_dic_keys)
        param_res = t.parse_table_dicts(col_nos, kics, type_arr, params)
        return param_res

    def get_periodic_or_not(self, kics):
        """
        determines if star is periodic or not my McQuillian paper
        """
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
        """
        uses kplr to get MAST parameters (kic10 mission)
        """
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
        """
        gets if star has neighbours or not

        :kics: list of kics to get results for
        :filter_params: mast-formatted parameter filters to determine what a neighbour is
        """
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

    def get_close_edges(self, kics, shape=(1070, 1132)):
        """
        gets distance of star to edge of detector by distance

        :kics: list of kics to get results for
        :shape: shape of detector image
        """
        reses = []
        rows_ks = ["Row_0", "Row_1", "Row_2", "Row_3"]
        cols_ks = ["Column_0", "Column_1", "Column_2", "Column_3"]
        mast_reses = self.get_mast_params(kics, rows_ks + cols_ks)
        for res in mast_reses:
            curr_params = {}
            rows = [res[k] if res[k] is not None else np.nan for k in rows_ks]
            cols = [res[k] if res[k] is not None else np.nan for k in cols_ks]
            edges = [np.nanmin(rows), abs(shape[0]-np.nanmax(rows)), \
                     np.nanmin(cols), abs(shape[0]-np.nanmax(rows))]
            curr_params["close_edges"] = edges
            reses.append(curr_params)
        return reses

    def get_close_labels(self, kics, min_distance=30, shape=(1070, 1132)):
        """
        gets label of side for star close to the edge of detector by min_distance
        use either get_close_edges or get_close_labels

        :kics: list of kics to get results for
        :min_distance: distance from edge of detector to be close
        :shape: shape of detector image
        """
        sides = ["Top", "Bottom", "Left", "Right"]
        reses = self.get_close_edges(kics, shape)
        for res in reses:
            edges = res["close_edges"]
            labels = [sides[i] for i in range(len(edges)) if edges[i] <= min_distance]
            curr_params = {"close_labels": labels}
            res.update(curr_params)
        return reses

    def get_lcs_times_uncerts(self, kics, lc_file):
        """
        gets calculated light curves, times, and uncertainties from our database
        """
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
        """
        gets star aperture from our database
        """
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
        """
        gets list of kepler quarters
        """
        t = table_api(lc_file, delimiter=" ", skip_rows=0, kic_col_no=None)
        times = t.get_nth_col(0, float)
        qs = t.get_nth_col(1, int)
        years = t.get_nth_col(2, int)
        curr_params = {"qs": qs, "years": years} # same pointer, unchanged data
        reses = [curr_params]*len(kics)
        return reses

    def _parse_var_params(self, string):
        parse_string = "[" + string + "]"
        literal_vals = ast.literal_eval(parse_string)
        return literal_vals

    def get_variable(self, kics, lc_file):
        """
        gets if star is variable or not with Chi-squared values from our database
        """
        t = table_api(lc_file, delimiter=",", skip_rows=1, kic_col_no=0)
        arrs = t.parse_table_arrs(range(1, 6), kics=kics, types=[str, str, float, float, str])
        reses = []
        for i, kic in enumerate(kics):
            curr_params = {}
            arr = arrs[i]
            if len(arr) == 0:
                curr_params["variable"], curr_params["curve_fit"], \
                    curr_params["var_chi2_best"], \
                    curr_params["var_bic_best"] = (np.nan for _ in range(4))
                logger.error("couldn't parse table for this kic: %s" % kic)
            else:
                curr_params["variable"] = 1 if arr[0] == "True" else 0
                curr_params["curve_fit"] = arr[1]
                curr_params["var_chi2_best"] = arr[2]
                curr_params["var_bic_best"] = arr[3]
                curr_params["var_res"] = self._parse_var_params(arr[4])
            reses.append(curr_params)
        return reses

if __name__ == "__main__":
    a = api()
    kics = ["757280", "757450"]
    b = a.get_mast_params(kics, ["g_KIS"])
    print b
