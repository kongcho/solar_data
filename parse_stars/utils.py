from settings import setup_logging
logger = setup_logging()

import csv
import funcsigs
import numpy as np

# checks if given kids have data from q1-17 kepler targets
# assume all kids are unique and all data exists
def get_existing_kics(arr1, arr2):
    list_kics = set(arr1).intersection(arr2)
    logger.info("done: %d kics" % len(list_kics))
    return list_kics

# checks if given kics is in file and that data exists for given columns
# WARNING: no way to separate split string by spaces and detect no data
def get_good_existing_kics(fin, list_kids, param_arr):
    list_params_kics = []
    exists_good_params = True
    with open(fin) as f:
        for line in f:
            curr_params = list(filter(None, line.strip().split(' ')))
            curr_params_kic = curr_params[0]
            for param_i in param_arr:
                if curr_params[param_i] == '':
                    exists_good_params = False
                    break
            if exists_good_params:
                list_params_kids.append(curr_param_kic)
                exists_good_params = True
    list_good_kids = set(list_kics).intersection(list_params_kics)
    logger.info("done: %d kids" % len(list_good_kics))
    return list_good_kics

# checks if given kics is in file and data exists for all columns
def get_good_kics(fin, list_kics):
    pass

# get all kids from several files
# assume all kids are unique from each file, assumes first line are fieldnames
def get_kics_files(list_files, sep=',', skip_rows=1):
    all_kics = []
    for files_i in range(len(list_files)):
        with open(list_files[files_i]) as f:
            for _ in range(skip_rows):
                next(f)
            reader = csv.reader(f, delimiter=sep, skipinitialspace=True)
            for row in reader:
                if row:
                    all_kics.append(row[0])
    logger.info("get_kics_files done: %d kics" % len(all_kics))
    return all_kics

# get list of kics from a file where kic is first of a column
def get_kics(fin, sep=',', skip_rows=0):
    all_kics = []
    with open(fin) as f:
        for _ in range(skip_rows):
            next(f)
        reader = csv.reader(f, delimiter=sep, skipinitialspace=True)
        for row in reader:
            all_kics.append(row[0])
    logger.info("done: %d kics" % len(all_kics))
    return all_kics

# get m kics for every nth kic
def get_nth_kics(fin, n, m, sep=',', skip_rows=0):
    kics = []
    start = 0
    if m > n:
        n, m = m, n
    with open(fin, 'r') as f:
        reader = csv.reader(f, delimiter=sep, skipinitialspace=True)
        for _ in range(skip_rows):
            next(f)
        for i, row in enumerate(reader, 1):
            if start < m:
                kics.append(row[0])
                start += 1
            else:
                if i % n == 0:
                    kics.append(row[0])
                    start = 0
    logger.info("done: %d kics" % len(kics))
    return kics

# converts nth row of file to array
def get_nth_row(fin, n, sep=','):
    arr = []
    with open(fin, 'r') as f:
        for _ in range(n-1):
            next(f)
        reader = csv.reader(f, delimiter=sep, skipinitialspace=True)
        for row in reader:
            arr.append(row)
    logger.info("done")
    return arr

# prompts warning if file exists
def does_path_exists(fin):
    if os.path.exists(fin):
        ans = raw_input("File " + fin + " already exists, proceed for all? (y/n) ")
        ans = ans.strip().lower()
        print(ans)
        if (ans != "y"):
            print("Quitting...")
            return True
    return False

# prints array to column in file, splits up files by kids_per_file
# returns length of array
def array_to_file(arr, fout, kids_per_file=9999, bypass_prompt=True):
    arr_i = 0
    total_kids = len(arr)
    output_files_i = 0
    no_of_files = total_kids//kids_per_file + 1
    remainder_of_files = total_kids%kids_per_file
    good_kids_filename = fin + "_" + str(output_files_i) + ".txt"

    if not bypass_prompt and does_path_exists(fout):
        return 1

    for output_files_i in range(number_of_files):
        good_kids_filename = fout + "_" + str(output_files_i) + ".txt"
        if (output_files_i == number_of_files - 1):
            number_kids_per_file = remainder_of_files
        with open(good_kids_filename, 'w') as write_f:
            writer = csv.writer(write_f, delimiter=',', lineterminator='\n')
            for i in range(arr_i, arr_i + kids_per_file):
                writer.writerow([arr[i]])
            arr_i = arr_i + kids_per_file
    logger.info(str(total_kids) + " entries in " + str(np_of_files) + " files")
    logger.info("done")
    return total_kids

# gets dimension of nest array
def get_dim(arr):
    if not type(arr[0]) == list:
        return 1
    return 1 + get_dim(arr[0])

# prints array to each row in filename, works for both 1d and 2d arrays
def simple_array_to_file(fout, arr):
    if type(arr) == np.ndarray:
        np.savetxt(fout, arr, delimiter=',', newline='\n')
    else:
        with open(fout, 'w') as f:
            writer = csv.writer(f, delimiter=',', lineterminator='\n')
            dim = get_dim(arr)
            if dim == 1:
                for row in arr:
                    writer.writerow([row])
            elif dim == 2:
                for row in arr:
                    writer.writerow(row)
            else:
                logger.error("can't support >2 dimensions")
    logger.info("done")
    return 0

# clips array based on maximum boundaries given
# is_max_bounds is True for upper bound, is False for minimum bound
def clip_array(coords, max_coords, is_max_bounds):
    length = len(coords)
    if length != len(max_coords):
        logger.error("clip_array dimensions must be same")
        return
    results = ()
    for i, coord in enumerate(coords):
        if is_max_bounds[i] == True:
            add = (coord,) if coord < max_coords[i] else (max_coords[i],)
        else:
            add = (coord,) if coord > max_coords[i] else (max_coords[i],)
        results += add
    return results

# calculates if elements of array satisfies boolean function at least n times
def is_n_bools(arr, n, bool_func):
    n_bools = False
    for i in arr:
        if bool_func(i):
            n -= 1
        if n == 0:
            n_bools = True
            break
    return n_bools

# formats array to print by separation
def format_arr(arr, sep="\t"):
    return sep.join(str(i) for i in arr)

# returns tuple of keys from a dictionary
def get_keys(dic):
    keys = ()
    for key in dic:
        keys += (key,)
    return keys

# helper, gets a subset of main_dic from subset of keys between both dictionaries
def get_union_dic(main_dic, secondary_dic):
    keys_main = get_keys(main_dic)
    keys_alt = get_keys(secondary_dic)
    sub_keys = set(keys_main).intersection(keys_alt)
    new_dic = {k: main_dic.get(k, None) for k in sub_keys}
    return new_dic

# gets the right kwargs for a function from a larger set of kwargs
def get_sub_kwargs(func, **kwargs):
    sig = funcsigs.signature(func)
    func_kwargs = get_union_dic(kwargs, sig.parameters)
    return func_kwargs

# builds array of strings and a counter as part of the string
def build_arr_n_names(name, n):
    arr = []
    max_digits = len(str(n))
    for i in range(n):
        arr.append(("%s_%0" + str(max_digits) + "d") % (name, i))
    return arr
