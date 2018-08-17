"""
API THAT WORKS FOR TABLE-BASED DATABASES
"""


from utils import does_path_exists
from settings import setup_logging, filename_stellar_params
logger = setup_logging()

import csv

class table_api(object):

    def __init__(self, table_file, delimiter=",", skip_rows=0, kic_col_no=None):
        self.filename = table_file
        self.delimiter = delimiter
        self.skip_rows = skip_rows
        self.kic_col_no = kic_col_no

    # converts nth col of file to array
    def get_nth_col(self, n, col_type=None):
        arr = []
        with open(self.filename, 'r') as f:
            for _ in range(self.skip_rows):
                next(f)
            reader = csv.reader(f, delimiter=self.delimiter, skipinitialspace=True)
            for row in reader:
                arr.append(row[n])
            if col_type is not None:
                try:
                    arr = map(col_type, arr)
                except Exception as e:
                    pass
        return arr

    # converts nth row of file to array
    def get_nth_row(self, n, types=None):
        with open(self.filename, 'r') as f:
            for _ in range(n-1):
                next(f)
            reader = csv.reader(f, delimiter=self.delimiter, skipinitialspace=True)
            arr = next(reader)
        if types is not None:
            for i, col_type in enumerate(types):
                try:
                    arr[i] = col_type(arr[i])
                except Exception as e:
                    pass
        return arr

    # function, gets columns indexes of given param names to parse table by column number
    def get_col_nos(self, params, field_names):
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

    def parse_table_arrs(self, col_nos, kics=None, types=None):
        completed_kics = [False]*len(kics) if kics is not None else [False]
        whole = []
        if self.kic_col_no is not None:
            try:
                col_nos.remove(self.kic_col_no)
            except:
                pass
            else:
                if types is not None:
                    del types[self.kic_col_no]
        with open(self.filename, "r") as f:
            for _ in range(self.skip_rows):
                next(f)
            r = csv.reader(f, delimiter=self.delimiter, skipinitialspace=True)
            for i, row in enumerate(r):
                if all(completed_kics):
                    return whole
                if kics is not None:
                    if row[self.kic_col_no] in kics:
                        completed_kics[kics.index(row[self.kic_col_no])] = True
                    else:
                        continue
                param_arr = []
                for j, no in enumerate(col_nos):
                    if types is not None:
                        try:
                            row[no] = types[j](row[no])
                        except Exception as e:
                            pass
                    param_arr.append(row[no])
                whole.append(param_arr)
        return whole

    # function, can parse through kepler_fov_search* and table_periodic tables
    def parse_table_dicts(self, col_nos, kics=None, types=None, params=None):
        completed_kics = [False]*len(kics) if kics is not None else [False]
        whole = []
        if params is None:
            params = map(str, col_nos)
        else:
            params = params
        with open(self.filename, "r") as f:
            for _ in range(self.skip_rows):
                next(f)
            r = csv.reader(f, delimiter=self.delimiter, skipinitialspace=True)
            for i, row in enumerate(r):
                if all(completed_kics):
                    return whole
                if kics is not None:
                    if row[self.kic_col_no] in kics:
                        completed_kics[kics.index(row[self.kic_col_no])] = True
                    else:
                        continue
                curr_dict = {}
                for j, no in enumerate(col_nos):
                    if types is not None:
                        try:
                            row[no] = types[j](row[no])
                        except Exception as e:
                            pass
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
