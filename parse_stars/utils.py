"""
USEFUL FUNCTIONS FOR GENERAL USAGE ACROSS ALL CODE
"""


from settings import setup_logging
logger = setup_logging()

import csv
import funcsigs
import numpy as np

def get_existing_kics(arr1, arr2):
    """
    checks subset of kics that exist within both arrays
    """
    list_kics = set(arr1).intersection(arr2)
    logger.info("done: %d kics" % len(list_kics))
    return list_kics


def get_nth_col(fin, n=0, sep=',', skip_rows=0):
    """
    gets nth column for all lines from given file
    """
    all_kics = []
    with open(fin) as f:
        for _ in range(skip_rows):
            next(f)
        reader = csv.reader(f, delimiter=sep, skipinitialspace=True)
        for row in reader:
            all_kics.append(row[n_col])
    logger.info("done: %d kics" % len(all_kics))
    return all_kics


def get_nth_kics(fin, n, m, nth_col=0, sep=',', skip_rows=0):
    """
    within nth column of table, gets m values for every n value
    """
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
                kics.append(row[nth_col])
                start += 1
            else:
                if i % n == 0:
                    kics.append(row[nth_col])
                    start = 0
    logger.info("done: %d kics" % len(kics))
    return kics


def does_path_exists(fin):
    """
    prompts warning if file exists
    """
    if os.path.exists(fin):
        ans = raw_input("File " + fin + " already exists, proceed for all? (y/n) ")
        ans = ans.strip().lower()
        print(ans)
        if (ans != "y"):
            print("Quitting...")
            return True
    return False


def array_to_file(arr, fout, n=9999, bypass_prompt=True):
    """
    prints array to column in file, limits to n lines per file
    """
    arr_i = 0
    total_kids = len(arr)
    output_files_i = 0
    no_of_files = total_kids//n + 1
    remainder_of_files = total_kids%n
    good_kids_filename = fin + "_" + str(output_files_i) + ".txt"

    if not bypass_prompt and does_path_exists(fout):
        return 1

    for output_files_i in range(number_of_files):
        good_kids_filename = fout + "_" + str(output_files_i) + ".txt"
        if (output_files_i == number_of_files - 1):
            number_kids_per_file = remainder_of_files
        with open(good_kids_filename, 'w') as write_f:
            writer = csv.writer(write_f, delimiter=',', lineterminator='\n')
            for i in range(arr_i, arr_i + n):
                writer.writerow([arr[i]])
            arr_i = arr_i + n
    logger.info(str(total_kids) + " entries in " + str(np_of_files) + " files")
    logger.info("done")
    return 0


def get_dim(arr):
    """
    gets dimension of nested array
    """
    if not type(arr[0]) == list:
        return 1
    return 1 + get_dim(arr[0])


def simple_array_to_file(fout, arr):
    """
    prints array to each row in filename, works for both 1d and 2d arrays
    """
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


def clip_array(coords, max_coords, is_max_bounds):
    """
    clips array values based on minimum/maximum boundaries given

    :max_coords: maximum boundaries of array, must match coords dimensions
    :is_max_bounds: is True for maximum/upper bounds, False for minimum bounds
    """
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


def is_n_bools(arr, n, bool_func):
    """
    calculates if elements of array satisfies boolean function at least n times
    """
    n_bools = False
    for i in arr:
        if bool_func(i):
            n -= 1
        if n == 0:
            n_bools = True
            break
    return n_bools


def format_arr(arr, sep="\t"):
    """
    formats array as string to print by separation
    """
    return sep.join(str(i) for i in arr)


def get_keys(dic):
    """
    return tuple of keys from a dictionary
    """
    keys = ()
    for key in dic:
        keys += (key,)
    return keys


def get_union_dic(main_dic, secondary_dic):
    """
    helper
    gets a subset of main dictionary from subset of keys between both dictionaries
    """
    keys_main = get_keys(main_dic)
    keys_alt = get_keys(secondary_dic)
    sub_keys = set(keys_main).intersection(keys_alt)
    new_dic = {k: main_dic.get(k, None) for k in sub_keys}
    return new_dic


def get_sub_kwargs(func, **kwargs):
    """
    helper to prevent errors
    gets the right optional arguments for a function from a larger set of arguments
    """
    sig = funcsigs.signature(func)
    func_kwargs = get_union_dic(kwargs, sig.parameters)
    return func_kwargs


def build_arr_n_names(name, n):
    """
    builds array of strings starting with a prefix and ends with a counter

    :name: prefix of string
    :n: number of strings to build
    """
    arr = []
    max_digits = len(str(n))
    for i in range(n):
        arr.append(("%s_%0" + str(max_digits) + "d") % (name, i))
    return arr
