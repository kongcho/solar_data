from settings import *
from utils import *
from mast import session, mast_api

import os
import numpy as np
import csv
import json

class new_star(object):

    def __init__(self, kic):
        table_file = "./tests/lc_data_new.out"
        self.kic = kic
        self._get_times(kic, table_file)
        self._get_lcs_and_uncerts(kic, table_file)

    def _make_target(self):
        self.target = run_photometry(self.kic)
        if self.target == 1:
            return self.target
        self.target.model_uncert()
        calculate_better_aperture(self.target, 0.001, 2, 0.7, 15)
        model_background(self.target, 0.2, 15)
        return 0

    def _get_lcs_times_uncerts(self, table_file):
        t = table_api(table_file, delimiter=",", skip_rows=1, kic_col_no=0)
        arr = t.parse_table_arrs(range(0, 108), kics=[self.kic], types=[float]*52)
        self.lcs = arr[:52]
        self.flux_uncert = arr[52:56]
        self.target_uncert = arr[56:108]
        self.times = t.get_nth_row(1)[1:53]
        return 0


class table_api(object):

    def __init__(self, table_file, delimiter=",", skip_rows=0, kic_col_no=None):
        self.filename = table_file
        self.delimiter = delimiter
        self.skip_rows = skip_rows
        self.kic_col_no = kic_col_no

    # converts nth row of file to array
    def get_nth_row(self, n):
        arr = []
        with open(self.filename, 'r') as f:
            for _ in range(n-1):
                next(f)
            reader = csv.reader(f, delimiter=self.delimiter, skipinitialspace=True)
            for row in reader:
                arr.append(row)
        return arr

    # function, gets columns indexes of given param names to parse table by column number
    def get_col_nos(self, params_want, field_names):
        col_nos = []
        for i, field in enumerate(field_names):
            if field in params:
                col_nos.append(i)
        return col_nos

    def get_col_nos_table(self, row_index, params):
        with open(self.filename, "r") as f:
            for _ in range(row_index):
                next(f)
            line = next(f)
        field_names = [x.strip() for x in line.split(self.delimiter)]
        return self.get_col_nos(params, field_names)

    def parse_table_arrs(self, col_nos, kics=None, types=None, n=100):
        completed_kics = [False]*len(kics) if kics is not None else [False]
        whole = []
        if self.kic_col_no is not None:
            try:
                col_nos.remove(self.kic_col_no)
            except:
                pass
        with open(self.filename, "r") as f:
            for _ in range(self.skip_rows):
                next(f)
            r = csv.reader(f, delimiter=self.delimiter, skipinitialspace=True)
            for i, row in enumerate(r):
                if i == n or all(completed_kics): #TBD
                    return whole #TBD
                if kics is not None:
                    if row[self.kic_col_no] in kics:
                        completed_kics[kics.index(row[self.kic_col_no])] = True
                    else:
                        continue
                param_arr = []
                for j, no in enumerate(col_nos):
                    if types is not None:
                        row[no] = types[j](row[no])
                    param_arr.append(row[no])
                whole.append(param_arr)
        return whole

    # function, can parse through kepler_fov_search* and table_periodic tables
    def parse_table_dicts(self, col_nos, kics=None, types=None, n=100):
        completed_kics = [False]*len(kics) if kics is not None else [False]
        whole = []
        if params is None:
            params = map(str, col_nos)
        else:
            params = params
        with open(self.filename, "r") as f:
            for _ in range(skip_rows):
                next(f)
            r = csv.reader(f, delimiter=self.delimiter, skipinitialspace=True)
            for i, row in enumerate(r):
                if i == n or all(completed_kics): #TBD
                    return whole #TBD
                if kics is not None:
                    if row[self.kic_col_no] in kics:
                        completed_kics[kics.index(row[self.kic_col_no])] = True
                    else:
                        continue
                curr_dict = {}
                for j, no in enumerate(col_nos):
                    if types is not None:
                        row[no] = types[j](row[no])
                    curr_dict[params[j]] = row[no]
                whole.append(curr_dict)
        return whole

    def filter_dic(self, dic, indexes):
        for key in dic:
            dic[key] = np.delete(dic[key], indexes)
        return dic

    # filters any dictionary result with a list of values for each param, with a params_dic
    # e.g. params_dic = {"Teff": [lambda x: x > 4]}
    def get_filtered_dic(self, info_dic, params_dic):
        good_bools = [True] * len(info_dic[info_dic.keys()[0]])
        for param in params_dic:
            good_bools = [True] * np.count_nonzero(good_bools)
            param_arr = np.array(info_dic[param])
            for bool_f in params_dic[param]:
                good_bools = np.logical_and(good_bools, bool_f(param_arr))
                good_indexes = np.where(good_bools == False)[0]
            filter_dic(info_dic, good_indexes)
        return info_dic

    def parse_fov_search_separate(self, fout_single, fout_batch):
        with open(self.filename, "r") as f, \
             open(fout_single, "wb") as fs, open(fout_batch, "wb") as fb:
            pass
            for curr_line in f:
                if curr_line[:10] == "Input line":
                    count = int()
            for _ in range(2):
                next(f)
        pass


class api(object):

    def __init__(self, kics, source=None):
        self.kic = kics
        self.sources = ["q1-17", "mast_kic10", "mast_target", "mast_table", \
                        "periodic", "nonperiodic", "kplr_target"]
        if source not in sources:
            logger.error("source is not defined in API")
            return None

        self.res = [{} for _ in kics]
        self.updated_dir = filename_stellar_params
        self.unperiodic_dir = filename_nonperiods
        self.periodic_dir = filename_periods
        self.mast_table_dir = filename_mast_table

    def get_params(self, kics, params):
        updated_params = ["teff", "logg", "metallicity", "rad", "mass" "rho", "dist", "av"]
        updated_pars, periodic_pars, mast_table_pars, \
            mast_pars, kplr_pars, incompletes = ([] for i in range(5))
        for param in params:
            if param in updated_params:
                updated_pars.append(param)
            elif param in self.updated_dic.keys():
                updated_pars.append(param)
            elif param in self.mast_dic.keys():
                mast_pars.append(param)
            elif param in self.periodic_dic.keys():
                periodic_pars.append(param)
            elif param in self.mast_table_dic.keys():
                mast_table_pars.append(param)
            elif param in self.kplr_dic.keys():
                kplr_pars.append(param)
            else:
                incompletes.append(param)
        self.get_updated_params(kics, q117_pars)
        self.get_periodic_or_not()

    def _update_params_arr(self, old_res, new_res):
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
        self.updated_dic = updated_dic
        col_name_arr, type_arr = self._format_params(self.updated_dic, params)
        t = table_api(self.updated_dir, " ", 0, 0)
        col_nos = t.get_col_nos(params, col_name_arr)
        param_res = t.parse_table_arrs(col_nos, kics, type_arr, None)
        self._update_params_arr(self.params, param_res)

    def get_periodic_or_not(self):
        periodic = get_kics(self.periodic_dir, ",", 1)
        unperiodics = get_kics(self.periodic_dir, ",", 1)
        if self.kic in periodic:
            self.params["periodic"] = True
        elif self.kic in unperiodics:
            self.params["periodic"] = False
        else:
            self.params["periodic"] = "Unsure"

    def get_periodic_params(self, params):
        self.periodic_heads = get_nth_row(self.periodic_dir, 1, ',')
        self.unperiodic_heads = get_nth_row(self.unperiodic_dir, 1, ',')
        self.mast_table_heads = get_nth_row(self.mast_table_dir, 1, ',')
        pass

    def get_mast_params(self, kics, params):
        basic_dic = {
            "kic_kepler_id": "8462852",
        }
        params_str = format_arr(params, ",")
        m = mast_api()
        hey = m.parse_target_params(basic_dic, ["kic_kepler_id", "kic_teff"])
        # hey = m.parse_target_params(basic_dic, params_str)
        print hey
        print hey.text

    def get_mast_table_params(self, params):
        pass

    def get_kplr_params(self, params):
        import kplr
        client = kplr.API()
        

    def get_neighbours_or_not(self, filter_params):
        return

def make_sound(duration=0.3, freq=440):
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    return 0

def main():
    logger.info("### starting ###")
    # np.set_printoptions(linewidth=1000, precision=4)


    # dics = parse_table_col([0, 1, 2], filename_periods, ",", 1, True, ["KID", "Teff", "logg"], [int, float, float], kics=None, kic_col_no=0, n=5)


    ###### NOWWWW ############


    # kic = "8462852"
    # import kplr
    # hey = kplr.API()
    # h2 = hey.star(int(kic))
    # print h2
    # h3 = hey.stars(kic_teff="5700..5800")
    # for h in h3:
    #     print h.kepid

    basic_dic = {
        # "ra": "12 46 11.086"
        # "kic_kepler_id": "8462852"
    }
    m = mast_api()
    hey = m._get_mission_params("kepler", "kepler_fov", basic_dic, maxrec=1)
    # hey = m.parse_target_params(basic_dic, ["kic_kepler_id", "kic_teff"])
    # hey = m.parse_target_params(basic_dic, params_str)
    print hey.text
    # print (hey.json())


    make_sound(0.3, 440)
    logger.info("### everything done ###")
    return 0

if __name__ == "__main__":
    os.sys.path.append(f3_location)
    from f3 import photometry

    from settings import setup_main_logging
    logger = setup_main_logging()
    main()


