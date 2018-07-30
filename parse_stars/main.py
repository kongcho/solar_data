from settings import *
from utils import *
from mast import session, mast_api

import os
import numpy as np
import csv
import json

class new_star(object):
    # field names for filename_stellar_params
    stellar_params = []

    def __init__(self, kic, table_file):
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

    def _get_lcs_and_uncerts(self, table_file):
        arr = parse_table_col(range(1, 53), table_file, ",", 1, False, [float]*52, kics=[self.kic])
        self.lcs = arr[:52]
        self.flux_uncert = arr[52:56]
        self.target_uncert = arr[56:108]
        return 0

    def _get_times(self, table_file):
        with open(table_file) as f:
            fields = f.readline().strip().split(",")
        self.times = fields[1:53]
        return 0

class table_api(object):
    def __init__(self, table_file):
        self.filename = table_file

    # function, can parse through kepler_fov_search* and table_periodic tables
    def parse_table_col(self, col_nos, delimiter=",", skip_rows=0, is_dic_not_arr=True, \
                        params=None, types=None, kics=None, kic_col_no=0, n=100):
        counter = 0 #TBD
        completed_kics = [False]*len(kics) if kics is not None else [False]
        if params is None:
            params = map(str, col_nos)
        if is_dic_not_arr:
            whole = {}
            for param in params:
                whole[param] = []
        else:
            whole = []
            if kic_col_no is not None:
                try:
                    col_nos.remove(kic_col_no)
                except:
                    pass
        with open(self.filename, "r") as f:
            for _ in range(skip_rows):
                next(f)
            r = csv.reader(f, delimiter=delimiter, skipinitialspace=True)
            for i, row in enumerate(r):
                counter += 1 #TBD
                if counter == n: #TBD
                    return whole #TBD
                if all(completed_kics):
                    return whole
                if kics is not None:
                    if row[kic_col_no] in kics:
                        completed_kics[kics.index(row[kic_col_no])] = True
                    else:
                        continue
                if not is_dic_not_arr:
                    param_arr = []
                for j, no in enumerate(col_nos):
                    if types is not None:
                        row[no] = types[j](row[no])
                    if is_dic_not_arr:
                        whole[params[j]].append(row[no])
                    else:
                        param_arr.append(row[no])
                if not is_dic_not_arr:
                    whole.append(param_arr)
        return whole

    def filter_dic(self, dic, indexes):
        for key in dic:
            dic[key] = np.delete(dic[key], indexes)
        return dic

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

    def parse_fov_search_all():
        pass

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

    # function, get column index by given param names, result is used as input to parse_table_col
    def get_col_nos(params, table_file, delimiter, row_index):
        col_nos = []
        with open(table_file, "r") as f:
            for _ in range(row_index):
                next(f)
            line = next(f)
        field_names = [x.strip() for x in line.split(delimiter)]
        for i, field in enumerate(field_names):
            if field in params:
                col_nos.append(i)
        return col_nos

def get_param_table_wrap(self, kic, params):
    # normal table
    table_1 = filename_periods
    col_nos = get_col_nos(params, table_1, ",", 0)
    params_dic = parse_table_col(col_nos, table_1, ",", 1, True, params, kics=[kic])
    return params_dic

def make_sound(duration=0.3, freq=440):
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    return 0

def main():
    logger.info("### starting ###")
    # np.set_printoptions(linewidth=1000, precision=4)

    # kics = get_kics("./data/table4.dat", " ", 0)
    # print kics[14429]
    # print kics[41156]
    # print kics[67908]
    # print kics[95041]
    # print kics[122011]
    # print kics[148552]
    # print kics[175500]


    # dics = parse_table_col([0, 1, 2], filename_periods, ",", 1, True, ["KID", "Teff", "logg"], [int, float, float], kics=None, kic_col_no=0, n=5)


    ###### NOWWWW ############

    t1 = ["target_name", "t_max"]
    t2 = [{"paramName": "target_name",
           "values": ["COMET-67P-CHURYUMOV-GER-UPDATE"]
    }]
    # print get_mast_params(t1, t2)

    # kic = "8462852"
    # import kplr
    # hey = kplr.API()
    # h1 = hey.mast_request("data_search")
    # h2 = hey.star(int(kic))
    # print h2
    # h3 = hey.stars(kic_teff="5700..5800")
    # for h in h3:
    #     print h.kepid


    dic = {
        "target_name": "8462852",
        "kic_radius": "0.5"
    }
    dics = json.dumps(dic)
    m = mast_api()
    print m._parse_json_output("kepler", "kepler_fov", ["Master ID"], dics, maxrec=10)

    make_sound(0.3, 440)
    logger.info("### everything done ###")
    return 0

if __name__ == "__main__":
    os.sys.path.append(f3_location)
    from f3 import photometry

    from settings import setup_main_logging
    logger = setup_main_logging()
    main()


