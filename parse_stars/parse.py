from utils import does_path_exists
from settings import setup_logging, filename_stellar_params
logger = setup_logging()

import csv

class table_api(object):
    """
    api for table-based databases

    :table_file: file path of table file
    :delimiter: delimiter between columns
    :skip_rows: number of header rows to skip before data
    :kic_col_no: primary key to use to select rows from
    :lineend: line terminator string
    """
    def __init__(self, table_file, delimiter=",", skip_rows=0, kic_col_no=None, lineend=""):
        self.filename = table_file
        self.delimiter = delimiter
        self.lineend = lineend
        self.skip_rows = skip_rows
        self.kic_col_no = kic_col_no

    def get_nth_col(self, n, col_type=None):
        """
        converts nth column of file to an array

        :n: column index to get
        :col_type: if need to convert data to a data type

        :return: 1D array
        """
        arr = []
        with open(self.filename, 'r') as f:
            for _ in range(self.skip_rows):
                next(f)
            reader = csv.reader(f, delimiter=self.delimiter, skipinitialspace=True)
            for row in reader:
                row[-1] = row[-1].strip(self.lineend)
                arr.append(row[n])
            if col_type is not None:
                try:
                    arr = map(col_type, arr)
                except Exception as e:
                    pass
        return arr

    def get_nth_row(self, n, types=None):
        """
        converts nth row of file to an array

        :n: row index to get
        :types: array of data types to convert rows to, str if no conversion

        :return: 1D array
        """
        with open(self.filename, 'r') as f:
            for _ in range(n-1):
                next(f)
            reader = csv.reader(f, delimiter=self.delimiter, skipinitialspace=True)
            arr = next(reader)
            arr[-1] = arr[-1].strip(self.lineend)
        if types is not None:
            for i, col_type in enumerate(types):
                try:
                    arr[i] = col_type(arr[i])
                except Exception as e:
                    pass
        return arr

    def get_col_nos(self, params, field_names):
        """
        gets column indexes of given column names

        :params: column names of data that you want
        :field_names: list of all field names of the database

        :return: list of column numbers of the given column names
        """
        col_nos = []
        for par in params:
            for i, field in enumerate(field_names):
                if field == par:
                    col_nos.append(i)
        return col_nos

    def get_col_nos_table(self, row_index, params):
        """
        gets column indexes based on database field names on given row

        :row_index: index with all field headers you need
        :params: column names of data that you want

        :return: list of column numbers of given column names
        """
        with open(self.filename, "r") as f:
            for _ in range(row_index):
                next(f)
            line = next(f).strip(self.lineend)
        field_names = [x.strip() for x in line.split(self.delimiter)]
        return self.get_col_nos(params, field_names)

    def parse_table_arrs(self, col_nos, kics=None, types=None):
        """
        gets sub-table of table given column numbers

        :col_nos: column number wanted
        :kics (optional): extract data only of the primary key matched by keys in this array
        :types (optional): convert data of extracted columns into these data types

        :return: array of arrays of wanted columns, matching primary key
        """
        completed_kics = [False]*len(kics) if kics is not None else [False]
        whole = [[]]*len(kics) if kics is not None else []
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
                row[-1] = row[-1].strip(self.lineend)
                if all(completed_kics):
                    return whole
                if kics is not None:
                    if row[self.kic_col_no] in kics:
                        whole_index = kics.index(row[self.kic_col_no])
                        completed_kics[whole_index] = True
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
                if kics is not None:
                    whole[whole_index] = param_arr
                else:
                    whole.append(param_arr)
        return whole

    def parse_table_dicts(self, col_nos, kics=None, types=None, params=None):
        """
        gets array of dictionary of table given column numbers

        :col_nos: column number wanted
        :kics (optional): extract data only of the primary key matched by keys in this array
        :types (optional): convert data of extracted columns into these data types
        :params (optional): array of field names of given columns

        :return: array of dictionary of wanted columns, matching primary key
        """
        completed_kics = [False]*len(kics) if kics is not None else [False]
        whole = [{}]*len(kics) if kics is not None else []
        if self.kic_col_no is not None:
            try:
                col_nos.remove(self.kic_col_no)
            except:
                pass
            else:
                if types is not None:
                    del types[self.kic_col_no]
        params = map(str, col_nos) if params is None else params
        with open(self.filename, "r") as f:
            for _ in range(self.skip_rows):
                next(f)
            r = csv.reader(f, delimiter=self.delimiter, skipinitialspace=True)
            for i, row in enumerate(r):
                row[-1] = row[-1].strip(self.lineend)
                if all(completed_kics):
                    return whole
                if kics is not None:
                    if row[self.kic_col_no] in kics:
                        whole_index = kics.index(row[self.kic_col_no])
                        completed_kics[whole_index] = True
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
                if kics is not None:
                    whole[whole_index] = curr_dict
                else:
                    whole.append(curr_dict)
        return whole


    def filter_dic(self, dic, indexes):
        for key in dic:
            dic[key] = np.delete(dic[key], indexes)
        return dic


    def get_filtered_dic(self, info_dic, params_dic):
        """
        filters any dictionary result with a list of values for each param, with a params_dic
        e.g. params_dic = {"Teff": [lambda x: x > 4]}

        :info_dic: dictionary of parameters (keys) and list of values (value) to filter through
        :params_dic: dictionary of parameter name and list of filters to apply to those params

        :return: dictionary of parameters and corresponding filtered list of values
        """
        good_bools = [True] * len(info_dic[info_dic.keys()[0]])
        for param in params_dic:
            good_bools = [True] * np.count_nonzero(good_bools)
            param_arr = np.array(info_dic[param])
            for bool_f in params_dic[param]:
                good_bools = np.logical_and(good_bools, bool_f(param_arr))
                good_indexes = np.where(good_bools == False)[0]
            filter_dic(info_dic, good_indexes)
        return info_dic
