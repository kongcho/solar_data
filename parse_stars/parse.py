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

    # converts nth row of file to array
    def get_nth_row(self, n):
        with open(self.filename, 'r') as f:
            for _ in range(n-1):
                next(f)
            reader = csv.reader(f, delimiter=self.delimiter, skipinitialspace=True)
            arr = next(reader)
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


### MAST TABLE PARSING


"""
- kepler_fov_search
  - has a lotttt of info shit
  - ugly af
  - can tell u if u have bright neighbours or not
  - ",", skip 2
"""

# prints one dict of kid, kepmag, angsep to csv file
def dict_to_file(fout, dicts, keys=None, bypass_prompt=True):
    if not bypass_prompt and does_path_exists(fout):
        return 1
    if keys == None:
        keys = [key for key in dicts[0]][1:]
    with open(fout, 'w') as write_f:
        for key_i in range(len(keys) - 1):
            write_f.write([keys[key_i]] + ",")
            write_f.write([keys[-1]] + "\n")
        for i in range(len(dicts)):
            write_f.write(dicts[i]["Kepler_ID"] + ",")
            for key_i in range(len(keys) - 1):
                write_f.write(dicts[i][keys[key_i]] + ",")
            write_f.write(dicts[i][keys[-1]] + "\n")
    return 0

# prints a dict of a kic and its neighbours to csv file
def dicts_to_file(filename, dicts, keys=None, bypass_prompt=True):
    if not bypass_prompt and does_path_exists(filename):
        return 1
    if keys == None:
        keys = [key for key in dicts[0]][1:]
    with open(filename, 'w') as write_f:
        for i in range(len(dicts)):
            write_f.write("input: " + dicts[i]["Kepler_ID"] + "\n")
            for kids in arr[i]:
                write_f.write(dicts[i]["Kepler_ID"] + ",")
            for key_i in range(len(keys) - 1):
                write_f.write(dicts[i][keys[key_i]] + ",")
            write_f.write(dicts[i][keys[-1]] + "\n")
    return 0

# helper for remove_bright_neighbours_together()
def element_is_not_in_list(arr, n):
    if len(arr) == 0:
        return True
    for i in range(len(arr)):
        if n == arr[i]:
            return False
    return True

# outputs separate files for stars with no neighbours and with neighbours
def remove_bright_neighbours(folder, fout_prefix, difference_max=2.0):
    single_kids = []
    batch_kids = []
    curr_dict = {}
    test_dict = {}
    batch_dict = {}
    kid_done = False
    kepmag_done = False
    angsep_done = False
    is_first_entry = True
    target_kid_has_data = True
    not_passed_last_entry = True
    fout_single = single_parsed_kids_filename
    fout_batch = batch_parsed_kids_filename

    input_files = sorted([filename for filename in os.listdir(folder) \
        if filename.startswith(kepmag_file_prefix)])

    for input_files_i in range(len(input_files)):
        with open(input_files[input_files_i]) as input_f:
            for line in input_f:
                curr_line = line.strip()
                if curr_line == "":
                    break
                if curr_line[0:10] == "Input line": #id line
                    if curr_line[10:13] == " 1:": # line under input 1 is labels
                        fieldnames = input_f.readline().strip().split(',')
                        for fields_i in range(len(fieldnames)):
                            if fieldnames[fields_i] == "Kepler_ID":
                                kid_col_no = fields_i
                                kid_done = True
                            if fieldnames[fields_i] == "kepmag":
                                kepmag_col_no = fields_i
                                kepmag_done = True
                            if fieldnames[fields_i] == "Ang Sep (')":
                                angsep_col_no = fields_i
                                angsep_done = True
                            # don't iterate through fieldnames unnecessarily
                            if kid_done and kepmag_done and angsep_done:
                                break
                        input_f.readline() #types, useless line
                        target_kid_has_data = True
                        curr_data = input_f.readline().strip().split(',')
                        curr_kid = curr_data[kid_col_no]
                        if curr_data[kepmag_col_no] == "" or curr_data[angsep_col_no] == "":
                            target_kid_has_data = False
                            continue
                        curr_kepmag = float(curr_data[kepmag_col_no])
                        curr_angsep = curr_data[angsep_col_no]
                        curr_dict = {"kepmag": curr_kepmag, "Ang Sep (')": curr_angsep}
                        continue
                    if is_first_entry: # previous star had no neighbours
                        curr_dict["Kepler_ID"] = curr_kid
                        single_kids.append(curr_dict)
                    else: # previously was a star with neighbours
                        batch_kids.append(batch_dict)
                        is_first_entry = True
                    target_kid_has_data = True
                    curr_data = input_f.readline().strip().split(',')
                    curr_kid = curr_data[kid_col_no]
                    if curr_data[kepmag_col_no] == "" or curr_data[angsep_col_no] == "":
                        target_kid_has_data = False
                        continue
                    index += 1
                    curr_kepmag = float(curr_data[kepmag_col_no])
                    curr_angsep = curr_data[angsep_col_no]
                    curr_dict = {"kepmag": str(curr_kepmag), "Ang Sep (')": curr_angsep}
                else:
                    if not target_kid_has_data:
                        continue
                    test_data = curr_line.split(',')
                    test_kid = test_data[kid_col_no]
                    if test_data[kepmag_col_no] == "" or test_data[angsep_col_no] == "":
                        continue
                    test_kepmag = float(test_data[kepmag_col_no])
                    test_angsep = test_data[angsep_col_no]
                    if abs(curr_kepmag - test_kepmag) <= difference_max:
                        test_dict = {"kepmag": str(test_kepmag), "Ang Sep (')": test_angsep}
                        if is_first_entry: # need to intialise dictionary
                            batch_dict = {"Kepler_ID": curr_kid, curr_kid: curr_dict, \
                                          test_kid: test_dict}
                            is_first_entry = False
                        else:
                            batch_dict[test_kid] = test_dict
            if not_passed_last_entry:
                if is_first_entry:
                    curr_dict["Kepler_ID"] = curr_kid
                    single_kids.append(curr_dict)
                else:
                    batch_kids.append(batch_dict)
                not_passed_last_entry = False
    dict_to_file(fout_single, single_kids)
    dicts_to_file(fout_batch, batch_kids)
    logger.info("printed " + str(len(single_kids)) + " kids with no neighbours")
    logger.info("printed " + str(len(batch_kids)) + " kids with neighbours")
    logger.info("remove_bright_neighbours_separate done")
    return 0

# TODO: function
def get_table_params(kics, params, fout, table_file=filename_stellar_params):
    if "Kepler_ID" not in params:
        params = ["Kepler_ID"] + params
    params_list = []
    curr_dicts = []
    with open(table_file) as input_f:
        curr_line = input_f.readline().strip()
        if curr_line[:13] == "Input line 1:": #id line
            fieldnames = input_f.readline().strip().split(',')
            input_f.readline() #types, useless line
            for f_i, fieldname in enumerate(fieldnames):
                if len(params_list) == len(params):
                    break
                for param in params:
                    if fieldname == param:
                        params_list.append((param, f_i))
            if len(params_list) != len(params):
                logger.error("Error: not all params found")
                return 1
        else:
            logger.error("Error: file doesn't have right format")
        for line in input_f:
            if line[:10] == "Input line":
                continue
            curr_dict = {}
            curr_data = line.strip().split(',')
            for par, par_i in params_list:
                curr_dict[par] = curr_data[par_i]
            curr_dicts.append(curr_dict)
    dict_to_file(fout, curr_dicts)
    if len(kics) != len(curr_dicts):
        logger.error("Error: not all kics are processed")
    logger.debug("get_table_params done")
    return 0

# TODO: function
def is_faint_table(target, min_kepmag=15, table_file=filename_stellar_params):
    with open(table_file) as input_f:
        curr_line = input_f.readline().strip()
        if curr_line[:13] == "Input line 1:": #id line
            fieldnames = input_f.readline().strip().split(',')
            input_f.readline() #types, useless line
            for f_i, fieldname in enumerate(fieldnames):
                if fieldname == "Kepler_ID":
                    kid_col_no = f_i
                if fieldname == "kepmag":
                    kepmag_col_no = f_i
        if curr_line[kid_col_no] == target:
            return curr_line[kepmag_col_no] > min_kepmag
        for line in input_f:
            if line[:10] == "Input line":
                continue
            curr_data = line.strip().split(',')
            if curr_data[kid_col_no] == target:
                return curr_data[kepmag_col_no] > min_kepmag
    logger.debug("get_table_params done")
    return False
