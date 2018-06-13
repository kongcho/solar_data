import csv
import sys
import os
import logging
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import lightkurve as lk
from math import sqrt
from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

from utils import *

## Setup
f3_location = "/home/user/Desktop/astrolab/solar_data/parse_stars/f3"
results_folder = "./results/"
output_folder = "./tests/"

## Input Files to change
filename_periods = "./data/Table_Periodic.txt"
filename_nonperiods = "./data/Table_Non_Periodic.txt"
list_filenames = [filename_periods]
stellar_param_filename = "./data/table4.dat" #KIC = 0, Teff, logg, Fe/H

kepmag_file_prefix = "./data/kepler_fov_search"

## Output Files
file_name = "good_kids"

output_file = output_folder + file_name + "_plot.pdf"
log_file = output_folder + file_name + ".log"
parsed_kids_filename = results_folder + file_name + "_parsed.txt"
single_parsed_kids_filename = results_folder + file_name + "_single.txt"
batch_parsed_kids_filename = results_folder + file_name + "_batch.txt"

targets_file = single_parsed_kids_filename
targets_files = [targets_file]

def set_filenames_based_on_folders(data_folder="./data/"):
    targets_files_temp = [data_folder + x.split('.')[0] + ".txt"
                          for x in os.listdir(data_folder)
                          if x.endswith(".txt") and "single" in x]
    if len(targets_files) != 0:
        file_name = targets_files_temp[0]
        targets_files = targets_files_temp

## Logging
# removes warning messages from kplr that is propagated to f3

format_str = '%(asctime)s [%(levelname)s]\t%(name)s: %(message)s' #%(module)s
formatter = logging.Formatter(format_str)

logging.basicConfig(stream=sys.stderr, level=logging.ERROR, format=format_str)
root = logging.getLogger()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

fh = logging.FileHandler(log_file)
fh.setFormatter(formatter)
logger.addHandler(fh)

## Functions

# checks if given kids have data from q1-17 kepler targets
# assume all kids are unique and all data exists
def get_existing_kids(param_filename, list_kids):
    list_existing_kids = []
    with open(param_filename) as f_param:
        for line in f_param:
            curr_param_kid = line.strip().split(' ')[0]
            for kids_i in range(len(list_kids)):
                if curr_param_kid == list_kids[kids_i]:
                    list_existing_kids.append(curr_param_kid)
    logger.info("get_existing_kids done")
    return list_existing_kids

# checks if specific parameters exists
# WARNING: no way to separate split string by spaces and detect no data
def get_good_kids(param_filename, list_kids, param_arr):
    list_good_kids = []
    exists_good_params = True
    with open(param_filename) as f_param:
        for line in f_param:
            curr_params = list(filter(None, line.strip().split(' ')))
            curr_param_kid = curr_params[0]
            for kids_i in range(len(list_kids)):
                if curr_param_kid == list_kids[kids_i]:
                    for param_i in param_arr:
                        if curr_params[param_i] == '':
                            exists_good_params = False
                            break
                    if exists_good_params:
                        list_good_kids.append(curr_param_kid)
                    exists_good_params = True
    logger.info("get_good_kids done")
    return list_good_kids

#///////////////////

# get all kids from several files
# assume all kids are unique from each file, assumes first line are fieldnames
def get_kics_files(list_files):
    kid_periods = []
    for files_i in range(len(list_files)):
        with open(list_files[files_i]) as f_periods:
            next(f_periods)
            reader = csv.reader(f_periods)
            for row in reader:
                if row:
                    kid_periods.append(row[0])
    logger.info("get_kics_files done")
    return(kid_periods)

# get list of kics from a file where kic is first of a column
def get_kics(filename):
    all_kics = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            all_kics.append(row[0])
    logger.info("get_kics done")
    return all_kics

# get m kics for every nth kic
def get_nth_kic(fin, n, m):
    kics = []
    start = 0
    if m > n:
        n, m = m, n
    with open(fin, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, 1):
            if start < m:
                kics.append(row[0])
                start += 1
            else:
                if i % n == 0:
                    kics.append(row[0])
                    start = 0
    return kics

# prompts warning if file exists
def does_path_exists(filename):
    if os.path.exists(filename):
        ans = raw_input("File " + filename + \
                        " already exists, proceed for all? (y/n) ")
        ans = ans.strip().lower()
        print(ans)
        if (ans != "y"):
            print("Quitting...")
            return True
    return False

#//////////////////////////

# prints array to column in file, splits up files by kids_per_file
# returns length of array
def array_to_file(arr, filename, kids_per_file=9999):
    arr_i = 0
    total_kids = len(arr)
    output_files_i = 0
    no_of_files = total_kids//kids_per_file + 1
    remainder_of_files = total_kids%kids_per_file
    good_kids_filename = filename + "_" + str(output_files_i) + ".txt"

    if does_path_exists(filename):
        return 1

    for output_files_i in range(number_of_files):
        good_kids_filename = file_name + "_" + str(output_files_i) + ".txt"
        if (output_files_i == number_of_files - 1):
            number_kids_per_file = remainder_of_files
        with open(good_kids_filename, 'w') as write_f:
            writer = csv.writer(write_f, delimiter=',', lineterminator='\n')
            for i in range(arr_i, arr_i + kids_per_file):
                writer.writerow([arr[i]])
            arr_i = arr_i + kids_per_file
    logger.info(str(total_kids) + " entries in " + str(np_of_files) + " files")
    logger.info("array_to_file done")
    return total_kids

def does_path_exists(filename):
    if os.path.exists(filename):
        ans = raw_input("File " + filename + \
                        " already exists, proceed for all? (y/n) ")
        ans = ans.strip().lower()
        print(ans)
        if (ans != "y"):
            print("Quitting...")
            return True
    return False

# prints array to each row in filename (good for kics)
def simple_array_to_file(filename, arr):
    with open(filename, 'w') as write_f:
        writer = csv.writer(write_f, delimiter=',', lineterminator='\n')
        for i in range(len(arr)):
            writer.writerow([arr[i]])
    return 0

def get_dic_keys(dic):
    keys = [key for key in dic]
    return keys

# prints one dict of kid, kepmag, angsep to csv file
def dict_to_file(filename, dicts, keys=None):
    if does_path_exists(filename):
        return 1
    if keys == None:
        keys = get_dic_keys(dicts[0])[1:]
    with open(filename, 'w') as write_f:
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
def dicts_to_file(filename, dicts, keys=None):
    if does_path_exists(filename):
        return 1
    if keys == None:
        keys = get_dic_keys(dicts[0])[1:]
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

# removes stars from list with bright neighbours, and removes duplicates
# assumes that list of neighbour stars follows each target stars, and
#   list of input stars has same order and is same as processed stars
def remove_bright_neighbours_together(folder, filename_out, difference_max=2.0):
    all_kids = []
    kepmag_col_no = 1
    curr_id = -1
    count = 0
    # filename = parsed_kids_filename

    if does_path_exists(filename):
        return 1

    input_files = sorted([filename for filename in os.listdir(folder) \
        if filename.startswith(kepmag_file_prefix)])

    with open(filename_out, 'w') as output_f:
        for input_files_i in range(len(input_files)):
            with open(input_files[input_files_i]) as input_f:
                for line in input_f:
                    curr_line = line.strip()
                    if curr_line[0:10] == "Input line": #id line
                        if curr_line[10:13] == " 1:": # under input 1 is labels
                            fieldnames = input_f.readline().strip().split(',')
                            for fields_i in range(len(fieldnames)):
                                if fieldnames[fields_i] == "Kepler_ID":
                                    kid_col_no = fields_i
                                if fieldnames[fields_i] == "kepmag":
                                    kepmag_col_no = fields_i
                            input_f.readline() #types, useless line
                        curr_data = input_f.readline().strip().split(',')
                        curr_kid = int(curr_data[kid_col_no])
                        curr_kepmag = float(curr_data[kepmag_col_no])
                        if element_is_not_in_list(all_kids, curr_kid):
                            all_kids.append(curr_kid)
                            output_f.write(str(curr_kid) + "\n")
                            count += 1
                    else:
                        test_data = curr_line.split(',')
                        test_kid = int(test_data[kid_col_no])
                        test_kepmag = test_data[kepmag_col_no]
                        if test_kepmag == "":
                            continue
                        elif abs(curr_kepmag - float(test_kepmag)) \
                             <= difference_max:
                            if element_is_not_in_list(all_kids, test_kid):
                                all_kids.append(test_kid)
                                output_f.write(str(test_kid) + "\n")
                                count += 1
    logger.info("printed " + str(count) + " kids")
    logger.info("remove_bright_neighbours_together done")
    return 0

# doesn't remove duplicates
# outputs separate files for stars with no neighbours and with neighbours
def remove_bright_neighbours_separate(folder, fout_prefix, difference_max=2.0):
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
                        if curr_data[kepmag_col_no] == "" or \
                           curr_data[angsep_col_no] == "":
                            target_kid_has_data = False
                            continue
                        curr_kepmag = float(curr_data[kepmag_col_no])
                        curr_angsep = curr_data[angsep_col_no]
                        curr_dict = {"kepmag": curr_kepmag, \
                                     "Ang Sep (')": curr_angsep}
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
                    if curr_data[kepmag_col_no] == "" or \
                       curr_data[angsep_col_no] == "":
                        target_kid_has_data = False
                        continue
                    index += 1
                    curr_kepmag = float(curr_data[kepmag_col_no])
                    curr_angsep = curr_data[angsep_col_no]
                    curr_dict = {"kepmag": str(curr_kepmag), \
                                 "Ang Sep (')": curr_angsep}
                else:
                    if not target_kid_has_data:
                        continue
                    test_data = curr_line.split(',')
                    test_kid = test_data[kid_col_no]
                    if test_data[kepmag_col_no] == "" or \
                       test_data[angsep_col_no] == "":
                        continue
                    test_kepmag = float(test_data[kepmag_col_no])
                    test_angsep = test_data[angsep_col_no]
                    if abs(curr_kepmag - test_kepmag) <= difference_max:
                        test_dict = {"kepmag": str(test_kepmag), \
                                     "Ang Sep (')": test_angsep}
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

# helper function that determines 3-point pattern, but all quarters from the
#   same channel must have the same pattern
def is_more_or_less_all(target, quarters):
    is_strange = 0
    is_incr = False
    is_decr = False
    for indexes in quarters:
        for i in range(len(indexes) - 1):
            if target.obs_flux[indexes[i + 1]] > target.obs_flux[indexes[i]]:
                if is_decr:
                    is_decr = False
                    break
                is_incr = True
            elif target.obs_flux[indexes[i + 1]] < target.obs_flux[indexes[i]]:
                if is_incr:
                    is_incr = False
                    break
                is_decr = True
        if is_strange == 0:
            if is_incr:
                is_strange = 1
            elif is_decr:
                is_strange = -1
            else:
                return 0
        elif (is_strange == 1 and not is_incr) or \
             (is_strange == -1 and not is_decr):
            return 0
    return is_strange

# helper function for is_more_or_less
# returns are all but one element is nonzero in an array
def most_are_same(arr):
    counts = Counter(arr)
    for no, count in counts.items():
        if (count >= (len(arr) - 1)) and (no != 0):
            return True, no
    return False

# helper function that determines 3-point pattern
# all but one pattern from same channel must be the same
# returns -1 for downward, +1 for upward
def is_more_or_less(target, quarters):
    curr_ch = []
    is_incr = False
    is_decr = False
    for indexes in quarters:
        for i in range(len(indexes) - 1):
            if target.obs_flux[indexes[i + 1]] > target.obs_flux[indexes[i]]:
                if is_decr:
                    is_decr = False
                    break
                is_incr = True
            elif target.obs_flux[indexes[i + 1]] < target.obs_flux[indexes[i]]:
                if is_incr:
                    is_incr = False
                    break
                is_decr = True
        is_strange = 1 if is_incr else -1 if is_decr else 0
        curr_ch.append(is_strange)
    is_nontrivial = most_are_same(curr_ch)
    if is_nontrivial:
        return is_nontrivial[1]
    return 0

# boolean function: determines if 3-point pattern exists
# aperture is too large (many stars in aperture) or too small (psf going out)
def is_large_ap(target):
    golden = range(0, 8)
    blacks = [[8, 9], [20, 21, 22], [31, 32, 33], [43, 44, 45]]
    reds = [[10, 11, 12], [23, 24, 25], [34, 35, 36], [46, 47, 48]]
    # blues: 13, 14 on same day, ignoring 13
    blues = [[14, 15, 16], [26, 27], [37, 38, 39], [49, 50, 51]]
    greens = [[17, 18, 19], [28, 29, 30], [40, 41, 42]]
    channels = [blacks, reds, blues, greens]
    for channel in channels:
        if is_more_or_less(target, channel) != 0:
            logger.info("is_large_ap True")
            return True
    logger.info("is_large_ap False")
    return False

# helper function for has_(close)_peaks: is peak if greater than all neighbours
#   and is brighter than center peak by factor, assumes center peak = target
def is_peak(img, xi0j0, xi0j1, xi0j2, xi1j0, xi2j0, factor=0.75):
    center = img[len(img[0])//2][len(img)//2]
    min_bright = factor * center
    others = [xi0j1, xi0j2, xi1j0, xi2j0]
    booleans = [xi0j0 != 0
                , all(x != 0 for x in others)
                , all(x < xi0j0 for x in others)
                , all(x >= min_bright for x in others)
               ]
    return all(booleans)

# boolean function: determines if there's a peak within a given distance
#   around center point (target star)
def has_close_peaks(target, diff=7, min_factor=1):
    img = target.img
    len_x = len(img[0])
    len_y = len(img)
    c_i = len_x//2
    c_j = len_y//2
    # center_peak = target.img[c_i][c_j]
    if c_i <= diff:
        min_i = 1
        max_i = len_x - 1
    else:
        min_i = c_i - diff
        max_i = c_i + diff
    if c_j <= diff:
        min_j = 1
        max_j = len_y - 1
    else:
        min_j = c_j - diff
        max_j = c_j + diff
    for i in range(min_i, max_i):
        for j in range(min_j, max_j):
            if i-1 == 14 or j-1 == 14 or i+1 == 16 or j+1 == 16:
                continue
            if is_peak(img, img[i][j], img[i][j-1], img[i][j+1], \
                       img[i-1][j], img[i+1][j], min_factor):
                logger.info("has_close_peaks True")
                return True
    logger.info("has_close_peaks False")
    return False

# boolean function: determines if aperture has more than one bright peak
def has_peaks(target, min_factor=1):
    img = target.img
    # center_peak = img[len(img[0])//2][len(img)//2]
    for i in range(2, len(img) - 2):
        for j in range(2, len(img[i]) - 2):
            if i-1 == 14 or j-1 == 14 or i+1 == 16 or j+1 == 16:
                continue
            if is_peak(img, img[i][j], img[i][j-1], img[i][j+1], \
                       img[i-1][j], img[i+1][j], min_factor):
                logger.info("has_peaks True")
                return True
    logger.info("has_peaks False")
    return False

def is_faint(target, limit=5500000):
    c_i = len(target.img[0])//2
    c_j = len(target.img)//2
    c_i = 15
    c_j = 15
    is_faint = True
    stop = False
    for i in range(c_i - 1, c_i + 2):
        for j in range(c_i - 1, c_i + 2):
            if stop:
                break
            if target.img[i][j] != 0.0 and target.img[i][j] >= limit:
                is_faint = False
                stop = True
                break
    logger.info("is_faint done: %s" % is_faint)
    return is_faint

# boolean function: always passes every star, for testing
def fake_bool(target):
    logger.info("fake_bool done")
    return True

# creates plot for one target, assumes already have obs_flux, flux_uncert
def plot_data(target, count=0):
    fig = plt.figure(figsize=(11,8))
    gs.GridSpec(3,3)

    plt.subplot2grid((3,3), (1,2))
    plt.title(target.kic, fontsize=20)
    plt.imshow(target.img, interpolation='nearest', cmap='gray', \
               vmin=98000*52, vmax=104000*52)

    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    for i in range(4):
        g = np.where(target.qs == i)[0]
        plt.errorbar(target.times[g], target.obs_flux[g], \
                     yerr=target.flux_uncert[i], fmt=target.fmt[i])
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Relative Flux', fontsize=15)

    fig.text(7.5/8.5, 0.5/11., str(count + 1), ha='center', fontsize = 12)
    fig.tight_layout()

    logger.info("plot_data done")
    return target

# helper function for different functions
# runs find_other_sources under different parameters to change the aperture
def run_partial_photometry(target, image_region=15, edge_lim=0.015, \
                           min_val=5000, ntargets=100, extend_region_size=3, \
                           remove_excess=4, plot_window=15, plot_flag=False):

    target.find_other_sources(edge_lim, min_val, ntargets, extend_region_size, \
                              remove_excess, plot_flag, plot_window)

    target.data_for_target(do_roll=True, ignore_bright=0)

    jj, ii = target.center
    jj, ii = int(jj), int(ii)
    if ii - image_region <= 0 or jj - image_region <=0:
        logger.info("run_partial_photometry unsuccessful")
        return 1

    img = np.sum(((target.targets == 1)*target.postcard + \
                  (target.targets == 1)*100000)
                 [:,jj-image_region:jj+image_region, \
                  ii-image_region:ii+image_region], axis=0)
    setattr(photometry.star, 'img', img)

    logger.info("run_partial_photometry done")
    return target

# sets up photometry for a star and adds aperture to class
def run_photometry(targ, image_region=15, edge_lim=0.015, min_val=5000, \
                   ntargets=100, extend_region_size=3, remove_excess=4, \
                   plot_window=15, plot_flag=False):

    target = photometry.star(targ)
    target.make_postcard()

    return run_partial_photometry(target, image_region, edge_lim, min_val, \
                           ntargets, extend_region_size, remove_excess, \
                           plot_window, plot_flag)

# helper function for plot_targets that sets up photometry of a star
#   runs a photometry and tests a list of boolean functions on it
#   then creates a plot for it with plot_data
def tests_booleans(targ, boolean_funcs, count, \
                   edge_lim=0.015, min_val=5000, ntargets=100):

    target = run_photometry(targ, edge_lim=edge_lim, \
                            min_val=min_val, ntargets=ntargets)
    if target == 1:
        return 1
    for boolean in boolean_funcs:
        # TODO: not for picking out bad ones
        if not boolean(target):
            return 1
    logger.info("tests_booleans done")
    return plot_data(target, count)

# outputs dict of functions that finds faulty stars
#   and kics that fall in those functions
def get_boolean_stars(targets, boolean_funcs, \
                      edge_lim=0.015, min_val=500, ntargets=100):

    full_dict = {}
    full_dict["good"] = []
    for boolean_func in boolean_funcs:
        full_dict[boolean_func.__name__] = []
    for targ in targets:
        is_faulty = False
        target = run_photometry(targ, edge_lim=edge_lim, \
                                min_val=min_val, ntargets=ntargets)
        if target == 1:
            return 1
        for boolean in boolean_funcs:
            if boolean(target):
                full_dict[boolean.__name__].append(target)
                is_faulty = True
        if not is_faulty:
            full_dict["good"].append(target)
    logger.info("get_boolean_stars done")
    return full_dict

# plots list of targets to a filename if the boolean function is true
def plot_targets(filename, boolean_funcs, targets):
    filename = filename.rsplit(".", 1)[0]
    total = len(targets)
    count = 1
    parsed_targets = []
    if len(boolean_funcs) == 1:
        output_file = filename + "_" + str(boolean_funcs[0].__name__) + ".pdf"
    else:
        output_file = filename + "_bads.pdf"
    with PdfPages(output_file) as pdf:
        for targ in targets:
            logger.info("# " + targ)
            target = tests_booleans(targ, boolean_funcs, count)
            if target != 1:
                plt.gcf().text(4/8.5, 1/11., str(target.img[15][15]), ha='center', fontsize = 12)
                parsed_targets.append(target)
                pdf.savefig()
                plt.close()
                logger.info(str(count) + "\t" + targ + "\tplot_done")
                count += 1
    logger.info(str(count - 1) + " out of " + str(total) + " targets plotted")
    logger.info("plot_targets done")
    return parsed_targets

def monotonic_arr_new(arr, is_decreasing, diff_flux=0):
    new_arr = arr if is_decreasing else np.flip(arr, 0)
    diffs = np.diff(new_arr)
    length = range(len(diffs))
    for diff in zip(length, length[1:]):
        i, j = diff
        if diffs[i] >= diff_flux and diffs[j] >= diff_flux:
            return i+1 if is_decreasing else len(arr)-(j+1)
    return -1 if is_decreasing else 0

def monotonic_arr(arr, is_decreasing, diff_flux=0):
    new_arr = arr if is_decreasing else np.flip(arr, 0)
    diffs = np.diff(new_arr)
    for i, diff in enumerate(diffs):
        if diff >= diff_flux:
            return i+1 if is_decreasing else len(arr)-(i+1)
    return -1 if is_decreasing else 0

def remove_nonzeros(arr, is_before):
    length = arr.shape[0]
    zeros = np.where(arr <= 5)[0]
    first = zeros[0]
    last = zeros[-1]
    if is_before: new = np.append(np.zeros(first), arr[first:])
    else: new = np.append(arr[:last], np.zeros(length-last))
    return new

def improve_aperture_simple(img):
    len_x = img.shape[1]
    len_y = img.shape[0]
    c_j = len_x//2
    c_i = len_y//2
    for i in range(len_y):
        print(str(i) + " " + str(np.nonzero(img[i,:])[0]))
        img[:][i] = np.concatenate((remove_nonzeros(img[i,:(c_j-1)], True), \
                              img[i,(c_j-1):(c_j+1)], \
                              remove_nonzeros(img[i,(c_j+1):], False)))
    return img

def img_to_new_aperture(target, img):
    pass

def recalculate_aperture(target):
    ii, jj = target.center
    ii, jj = int(ii), int(jj)
    image_region = 15

    target.roll_best = np.zeros((4,2))
    for i in range(4):
        g = np.where(target.qs == i)[0]
        wh = np.where(target.times[g] > 54947)
        target.roll_best[i] = target.do_rolltest(g, wh)
    target.do_photometry()

    return 0

def improve_aperture(target, image_region=15):
    img = target.img
    len_x = img.shape[1]
    len_y = img.shape[0]
    c_j = len_x//2
    c_i = len_y//2
    ii, jj = target.center
    ii, jj = int(ii), int(jj)
    len_targ_y, len_targ_x = target.targets.shape

    # go through rows
    for i in range(len_y):
        targets_i = i+ii-image_region
        inc = monotonic_arr_new(img[i,:c_j], is_decreasing=False)
        if inc != 0:
            img[i,:inc] = 0
            target.targets[targets_i,:(inc+jj-image_region)] = 0
        dec = monotonic_arr_new(img[i,(c_j+1):], is_decreasing=True)
        if dec != -1:
            real_dec = c_j + 1 + dec
            img[i,real_dec:] = 0
            target.targets[targets_i,(real_dec+jj-image_region):] = 0

    # go through cols
    for j in range(len_x):
        targets_j = j+jj-image_region
        inc = monotonic_arr_new(img[:c_i,j], is_decreasing=False)
        if inc != 0:
            img[:inc,j] = 0
            target.targets[:(inc+ii-image_region),targets_j] = 0
        dec = monotonic_arr_new(img[(c_i+1):,j], is_decreasing=True)
        if dec != -1:
            real_dec = c_i + 1 + dec
            img[real_dec:,j]=0
            target.targets[(real_dec+ii-image_region):,targets_j] = 0

    recalculate_aperture(target)

    return img

def improve_aperture_mask(target, mask, image_region=15):
    img = target.img
    len_x = img.shape[1]
    len_y = img.shape[0]
    c_j = len_x//2
    c_i = len_y//2
    ii, jj = target.center
    ii, jj = int(ii), int(jj)
    len_targ_y, len_targ_x = target.targets.shape

    print mask

    i = 0
    j = 0
    for r, row in enumerate(mask):
        for c, x in enumerate(row):
            if x==0:
                i=c+ii-image_region
                j=r+jj-image_region
                img[r, c] = 0
                target.targets[i, j] = 0

    first = np.where(img>0, 1, 0)
    print np.subtract(mask, first)

    # go through rows
    for i in range(len_y):
        targets_i = i+ii-image_region
        inc = monotonic_arr_new(img[i,:c_j], is_decreasing=False)
        if inc != 0:
            img[i,:inc] = 0
            target.targets[targets_i,:(inc+jj-image_region)] = 0
        dec = monotonic_arr_new(img[i,(c_j+1):], is_decreasing=True)
        if dec != -1:
            real_dec = c_j + 1 + dec
            img[i,real_dec:] = 0
            target.targets[targets_i,(real_dec+jj-image_region):] = 0

    # go through cols
    for j in range(len_x):
        targets_j = j+jj-image_region
        inc = monotonic_arr_new(img[:c_i,j], is_decreasing=False)
        if inc != 0:
            img[:inc,j] = 0
            target.targets[:(inc+ii-image_region),targets_j] = 0
        dec = monotonic_arr_new(img[(c_i+1):,j], is_decreasing=True)
        if dec != -1:
            real_dec = c_i + 1 + dec
            img[real_dec:,j]=0
            target.targets[(real_dec+ii-image_region):,targets_j] = 0

    recalculate_aperture(target)

    second= np.where(img>0, 1, 0)
    print np.subtract(first, second)

    print "AHHHH"
    print np.subtract(mask, second)

    return img


def is_std_better_biggest(old_stds, stds):
    max_i = np.argmax(stds)
    return stds[max_i] <= old_stds[max_i]

def is_std_better_avg(old_stds, stds):
    return np.average(stds) <= np.average(old_stds)

# TODO: test x number of parameters
# runs through lists of different parameters, print flux plots and apertures,
#   then tests if it has only 1 peak, then takes out lowest stddev
def print_better_apertures(targ, boolean_func, edge_lim=0.015, min_val=5000, \
                           extend_region_size=3, remove_excess=4):

    target = photometry.star(targ)
    target.make_postcard()

    edge_lims = np.arange(edge_lim - 0.010, edge_lim + 0.025, 0.005)
    min_vals = np.arange(min_val - 2000, min_val + 2000, 500)
    region_sizes = np.arange(2, 5)
    excesses = np.arange(2, 6)

    test_vars = [edge_lims, min_vals, region_sizes, excesses]
    vals = [list(itertools.product(*test_vars))]

    run_partial_photometry(target, edge_lim=0.015, min_val=5000, \
                           extend_region_size=3, remove_excess=4, ntargets=100)

    old_stds = target.flux_uncert
    plot_data(target)

    with PdfPages(output_file) as pdf:
        for count, val in enumerate(vals, 1):
            res = {}
            run_partial_photometry(target, edge_lim=val[0], min_val=val[1], \
                                   extend_region_size=val[2], \
                                   remove_excess=val[3], ntargets=100)
            res["settings"] = "edge: " + str(val[0]) + " min: " + str(val[1]) + " region: " + str(val[2]) + " excess: " + str(val[3])
            res["boolean"] = boolean_func(target)
            res["is_avg"] = is_std_better_avg(old_stds, target.flux_uncert)
            res["is_most"] = is_std_better_biggest(old_stds, target.flux_uncert)
            res["has_peaks"] = has_close_peaks(target)
            results[val] = res
            plot_data(target, count)
            plt.gcf().text(4/8.5, 1/11., str(res), ha='center', fontsize = 11)
            pdf.savefig()
            plt.close()
    logger.info("get_better_apertures done")
    return

def print_best_apertures(targ, edge_lim=0.015, min_val=5000, \
                         extend_region_size=3, remove_excess=4, min_factor=0.7):
    fout = targ + "_plot.pdf"
    target = photometry.star(targ)
    target.make_postcard()

    best_params = (edge_lim, min_val, extend_region_size, remove_excess)
    edge_lims = np.arange(edge_lim - 0.010, edge_lim + 0.025, 0.01)
    min_vals = np.arange(min_val - 2000, min_val + 2000, 1000)
    region_sizes = np.arange(1, 3)
    excesses = np.arange(1, 4)

    single_results = []
    all_vals = []

    for v0, v1, v2, v3 in itertools.product(edge_lims, min_vals, region_sizes, excesses):
        all_vals.append((v0, v1, v2, v3))

    with PdfPages(targ + "_plot_1.pdf") as pdf, \
         PdfPages(targ + "_plot_2.pdf") as pdf2:
        if run_partial_photometry(target, edge_lim=0.015, min_val=5000, \
                              extend_region_size=3, remove_excess=4, \
                              ntargets=100) == 1:
            return 1

        best_uncert = target.flux_uncert
        plot_data(target)
        plt.gcf().text(4/8.5, 1/11., str(best_params), ha='center', fontsize = 11)
        pdf.savefig()
        pdf2.savefig()
        plt.close()

        for count, vals in enumerate(all_vals, 1):
            res = {}
            if run_partial_photometry(target, edge_lim=vals[0], min_val=vals[1], \
                                      extend_region_size=vals[2],
                                      remove_excess=vals[3], ntargets=100) == 1:
                continue
            res["settings"] = vals
            res["has_peaks"] = has_peaks(target, min_factor)
            res["is_avg"] = is_std_better_avg(best_uncert, target.flux_uncert)
            res["is_most"] = is_std_better_biggest(best_uncert, target.flux_uncert)
            plot_data(target)
            plt.gcf().text(4/8.5, 1/11., str(res), ha='center', fontsize = 11)
            pdf2.savefig()
            plt.close()
            if not res["has_peaks"]:
                single_results.append((np.average(target.flux_uncert), vals))

        if len(single_results) != 0:
            best_uncert, best_params = single_results[single_results.index(min(single_results))]

        if run_partial_photometry(target, edge_lim=best_params[0], \
                                  min_val=best_params[1], \
                                  extend_region_size=best_params[2], \
                                  remove_excess=best_params[3], \
                                  ntargets=100) == 1:
            return 1
        plot_data(target)
        plt.gcf().text(4/8.5, 1/11., str((best_params, best_uncert)), ha='center', fontsize = 11)
        pdf.savefig()
        pdf2.savefig()
        plt.close()
    logger.info("get_best_apertures done")
    return 0

def is_faint_table(target, min_kepmag=15, table_file=stellar_param_filename):
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

"""
Kepler_ID,Master ID,kepmag,Ang Sep ('),RA (J2000),
Dec (J2000),Data Availability,Seasons on CCD,Teff,Log_G,
E(B-V),g,r,J,GALEX NUV,
ID_IRT,J_IRT,gr,gi,gK,
gJ,GALEX FUVNUV,NUVg,Contamination season 0,Flux Fraction season 0,
Edge_Distance_0 (px),Module_0,Module_1,Module_2,Module_3,
Ug_KIS,gr_KIS,gi_KIS,BV_UBV,UB_UBV,
gJ_KIS_IRT,V_UBV,g_KIS,r_KIS,class_g_KIS,
Condition flag,ID_SDSS,u_SDSS,g_SDSS,r_SDSS,
i_SDSS,z_SDSS,ug_SDSS,gr_SDSS,gi_SDSS

"""

def get_table_params(kics, params, fout, table_file=stellar_param_filename):
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
                print(params)
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

def get_mast_params(target, params):
    data = target.mast_request("kepler", "Kepler_ID")
    print(data)
    return

def testing(targ):

    target = run_photometry(targ)
    if target == 1:
        return

    tar = target.target
    channel = [tar.params['Channel_0'], tar.params['Channel_1'],
               tar.params['Channel_2'], tar.params['Channel_3']]

    kepprf = lk.KeplerPRF(channel=channel[0], shape=(30, 30), column=15, row=15)
    prf = kepprf(flux=1000, center_col=30, center_row=30,scale_row=1, scale_col=1, rotation_angle=0)
    new = np.where(prf > 0.0005*np.max(prf), 1, 0)

    with PdfPages(targ + "_out.pdf") as pdf:
        plot_data(target)
        plt.gcf().text(4/8.5, 1/11., str(np.average(target.flux_uncert)), ha='center', fontsize = 11)
        pdf.savefig()
        improve_aperture_mask(target, new)
        plot_data(target)
        plt.gcf().text(4/8.5, 1/11., str(np.average(target.flux_uncert)), ha='center', fontsize = 11)
        pdf.savefig()
        plt.close()

    # print(target.times)
    # print("EHH")
    # print(target.obs_flux)
    # print "OIHFOIEF"
    # print target.flux_uncert

    logger.info("testing done")
    return target

def main():
    logger.info("### starting ###")
    ## gets good kics to be put into MAST
    # array_to_file(get_good_kids(stellar_param_filename, get_kics_files(list_filenames)))

    ## uses MAST results to get stars without neighbours
    # remove_bright_neighbours_separate()

    ## plots list of files with kics using f3
    # plot_targets(output_file, [fake_bool], ["893033"])
    # plot_targets(targets_file, [is_large_ap, has_peaks], get_kics(targets_file))

    ## TEMP

    ben_random = ["8462852"
                  , "3100219"
                  , "7771531"
                  # , "9595725"
                  # , "9654240"
                  # , "6691114"
                  # , "7109052"
                  # , "8043142"
                  # , "8544875"
                  # , "9152469"
                  # , "9762293"
                  # , "11447772"
                  # , "9210192"
                  # , "1161620"
                  ]

    ben_kics = ["2694810"
                , "3236788"
                , "3743810"
                , "4555566"
                , "4726114"
                , "5352687"
                , "5450764"
                , "6038355"
                , "6263983"
                , "6708110"
                , "7272437"
                , "7432092"
                , "7433192"
                , "7678238"
                , "8041424"
                , "8043142"
                , "8345997"
                , "8759594"
                , "8804069"
                , "9306271"
                , "10087863"
                , "10122937"
                , "11014223"
                , "11033434"
                , "11415049"
                , "11873617"
                , "12417799"
                ]

    bad_targs = ["893033"
                 , "1161620"
                 , "1162635"
                 , "1164102"
                 , "1293861"
                 , "1295289"
                 , "1430349"
                 , "1431060"
                 , "1162715"
                 , "1295069"
                 , "1433899"
                 ]

    # kics = get_kics("out03.txt")
    # kics = ["6542321", "2017224", "3745516", "2694810", "3853405", "4863614", "7691547", "8396113"]
    kics = ben_random

    # plot_targets(targets_file, [fake_bool], kics)

    # for i, targ in enumerate(kics):
    #     print(str(i) + "/" + str(len(kics)) + " new one!")
    #     print_best_apertures(targ)

    # li = []
    # with open("data/table3.dat", "r") as fin:
    #     for line in fin:
    #         data = line.split(" ")
    #         li.append(data[1])
    # with open("table3_out.txt", "w") as fout:
    #     wr = csv.writer(fout)
    #     for i in li:
    #         wr.writerow(i)

    ## TESTS
    # with open("out.txt", "w") as f:
    #     f.write("columns\n")
    #     wr = csv.writer(f)
    #     for kic in kics:
    #         target = testing(kic)
    #         wr.writerow(target.times + target.obs_flux)

    for kic in kics:
        testing(kic)
    logger.info("### everything done ###")
    return 0

if __name__ == "__main__" and __package__ is None:
    from os import sys, path
    sys.path.append(f3_location)
    from f3 import photometry
    main()
