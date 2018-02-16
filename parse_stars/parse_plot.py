import csv
import os
import logging
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from math import sqrt
from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

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

def set_filenames_based_on_folders():
    targets_files_temp = ["./data/" + x.split('.')[0] + ".txt"
                          for x in os.listdir("./data/")
                          if x.endswith(".txt") and "single" in x]
    if len(targets_files) != 0:
        file_name = targets_files_temp[0]
        targets_files = targets_files_temp

## Logging
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler(log_file)
handler.setFormatter(formatter)
logger.addHandler(handler)

# TODO: fix logging to stfout with proper format...?
#handler_stdout = logging.StreamHandler()
#handler_stdout.setFormatter(formatter)
#logger.addHandler(handler_stdout)


## Functions
# assume all kids are unique from each file
# assumes first line are fieldnames
def get_kids(list_files):
    kid_periods = []
    for files_i in range(len(list_files)):
        with open(list_files[files_i]) as f_periods:
            next(f_periods)
            reader = csv.reader(f_periods)
            for col in reader:
                if col:
                    kid_periods.append(col[0])
    logger.info("get_kids done")
    return(kid_periods)


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
# TODO: check if all parameters exist instead by checking if last element exists
def get_good_kids(param_filename, list_kids, param_arr):
    list_good_kids = []
    exists_good_params = True
    with open(param_filename) as f_param:
        for line in f_param:
            curr_params = list(filter(None,line.strip().split(' ')))
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


# prints array to column in file, splits up files by number_kids_per_file
# returns length of array
def array_to_file(arr, number_kids_per_file=9999):
    arr_i = 0
    total_kids = len(arr)
    output_files_i = 0
    number_of_files = total_kids//number_kids_per_file + 1
    remainder_of_files = total_kids%number_kids_per_file
    good_kids_filename = file_name + "_" + str(output_files_i) + ".txt"

    if os.path.exists(good_kids_filename):
        ans = input("File already exists, do you want to proceed for all? (y/n) ")
        ans = ans.strip().lower()
        if (ans != "y"):
            print("Quitting...")
            return 1

    for output_files_i in range(number_of_files):
        good_kids_filename = file_name + "_" + str(output_files_i) + ".txt"
        if (output_files_i == number_of_files - 1):
            number_kids_per_file = remainder_of_files
        with open(good_kids_filename, 'w') as write_f:
            writer = csv.writer(write_f, delimiter=',', lineterminator='\n')
            for i in range(arr_i, arr_i + number_kids_per_file):
                writer.writerow([arr[i]])
            arr_i = arr_i + number_kids_per_file
    logger.info(str(total_kids) + " entries in " + str(number_of_files) + " files")
    logger.info("array_to_file done")
    return total_kids

# prints array to each row in filename (good for kics)
def simple_array_to_file(filename, arr):
    if os.path.exists(filename):
        ans = input("File " + filename + " already exists, do you want to proceed for all? (y/n) ")
        ans = ans.strip().lower()
        if (ans != "y"):
            print("Quitting...")
            return 1
    with open(filename, 'w') as write_f:
        writer = csv.writer(write_f, delimiter=',', lineterminator='\n')
        for i in range(len(arr)):
            writer.writerow([arr[i]])
    return 0

# prints one dict of kid, kepmag, angsep to csv file
def dict_to_file(filename, arr):
    if os.path.exists(filename):
        ans = input("File " + filename + " already exists, do you want to proceed for all? (y/n) ")
        ans = ans.strip().lower()
        if (ans != "y"):
            print("Quitting...")
            return 1
    with open(filename, 'w') as write_f:
        for i in range(len(arr)):
            write_f.write(arr[i]["kid"] + ",")
            write_f.write(str(arr[i]["kepmag"]) + ",")
            write_f.write(arr[i]["angsep"] + "\n")
    return 0

# prints a dict of a kic and its neighbours to csv file
def dicts_to_file(filename, arr):
    if os.path.exists(filename):
        ans = input("File " + filename + " already exists, do you want to proceed for all? (y/n) ")
        ans = ans.strip().lower()
        if (ans != "y"):
            print("Quitting...")
            return 1
    with open(filename, 'w') as write_f:
        for i in range(len(arr)):
            write_f.write("input: " + arr[i]["kid"] + "\n")
            for kids in arr[i]:
                if kids != "kid":
                    write_f.write(kids + ",")
                    write_f.write(str(arr[i][kids]["kepmag"]) + ",")
                    write_f.write(arr[i][kids]["angsep"] + "\n")
    return 0


# helper for remove_bright_neighbours_together()
def element_is_not_in_list(arr, n):
    if len(arr) == 0:
        return True
    for i in range(len(arr)):
        if n == arr[i]:
            return False
    return True


# assumes that list of neighbour stars follows each target stars
# assumes list of input stars has same order and is same as processed stars
# removes duplicates
def remove_bright_neighbours_together(difference_max = 2.0):
    all_kids = []
    kepmag_col_no = 1
    curr_id = -1
    count = 0

    if os.path.exists(parsed_kids_filename):
        ans = input("File " + parsed_kids_filename + " already exists, do you want to proceed? (y/n) ")
        ans = ans.strip().lower()
        if (ans != "y"):
            print("Quitting...")
            return 1

    input_files = sorted([filename for filename in os.listdir('.') \
        if filename.startswith(kepmag_file_prefix)])

    with open(parsed_kids_filename, 'w') as output_f:
        for input_files_i in range(len(input_files)):
            with open(input_files[input_files_i]) as input_f:
                for line in input_f:
                    curr_line = line.strip()
                    if curr_line[0:10] == "Input line": #id line
                        if curr_line[10:13] == " 1:": # line under input 1 is labels
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
                        elif abs(curr_kepmag - float(test_kepmag)) <= 2.0:
                            if element_is_not_in_list(all_kids, test_kid):
                                all_kids.append(test_kid)
                                output_f.write(str(test_kid) + "\n")
                                count += 1
    logger.info("printed " + str(count) + " kids")
    logger.info("remove_bright_neighbours_together done")
    return 0


# doesn't remove duplicates
# outputs separate files for stars with no neighbours and with neighbours
def remove_bright_neighbours_separate(difference_max = 2.0):
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

    input_files = sorted([filename for filename in os.listdir('.') \
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
                            # don't need to iterate through fieldnames unnecessarily
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
                        curr_dict = {"kepmag": curr_kepmag, "angsep": curr_angsep}
                        continue
                    if is_first_entry: # previous star had no neighbours
                        curr_dict["kid"] = curr_kid
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
                    print("here " + curr_data[kepmag_col_no])
                    index += 1
                    curr_kepmag = float(curr_data[kepmag_col_no])
                    curr_angsep = curr_data[angsep_col_no]
                    curr_dict = {"kepmag": str(curr_kepmag), "angsep": curr_angsep}
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
                        test_dict = {"kepmag": str(test_kepmag), "angsep": test_angsep}
                        if is_first_entry: # need to intialise dictionary
                            batch_dict = {"kid": curr_kid, curr_kid: curr_dict, test_kid: test_dict}
                            is_first_entry = False
                        else:
                            batch_dict[test_kid] = test_dict
            if not_passed_last_entry:
                if is_first_entry:
                    curr_dict["kid"] = curr_kid
                    single_kids.append(curr_dict)
                else:
                    batch_kids.append(batch_dict)
                not_passed_last_entry = False
    dict_to_file(single_parsed_kids_filename, single_kids)
    dicts_to_file(batch_parsed_kids_filename, batch_kids)
    logger.info("printed " + str(len(single_kids)) + " kids with no neighbours")
    logger.info("printed " + str(len(batch_kids)) + " kids with neighbours")
    logger.info("remove_bright_neighbours_separate done")
    return 0

# creates plot for one target, assumes already have obs_flux, flux_uncert
def plot_data(targ, count=0, image_region=15, do_roll=True, ignore_bright=0):
    fig = plt.figure(figsize=(11,8))
    gs.GridSpec(3,3)

#    jj, ii = targ.center
#    jj, ii = int(jj), int(ii)

    plt.subplot2grid((3,3), (1,2))
    plt.title(targ.kic, fontsize=20)
#    img = np.sum(((targ.targets == 1)*targ.postcard + (targ.targets == 1)*100000)
#                 [:,jj-image_region:jj+image_region,ii-image_region:ii+image_region], axis=0)
    plt.imshow(targ.img, interpolation='nearest', cmap='gray', vmin=98000*52, vmax=104000*52)

    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    for i in range(4):
        g = np.where(targ.qs == i)[0]
        plt.errorbar(targ.times[g], targ.obs_flux[g], yerr=targ.flux_uncert[i], fmt=targ.fmt[i])
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Relative Flux', fontsize=15)

    fig.text(7.5/8.5, 0.5/11., str(count + 1), ha='center', fontsize = 12)
    fig.tight_layout()

    logger.info("plot_data done")
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
        elif (is_strange == 1 and not is_incr) or (is_strange == -1 and not is_decr):
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
    blues = [[14, 15, 16], [26, 27], [37, 38, 39], [49, 50, 51]] #[13, 14, 15, 16]
    greens = [[17, 18, 19], [28, 29, 30], [40, 41, 42]]
    channels = [blacks, reds, blues, greens]
    for channel in channels:
        if is_more_or_less(target, channel) != 0:
            logger.info("is_large_ap True")
            return True
    logger.info("is_large_ap False")
    return False

def is_peak(img, xi0j0, xi0j1, xi0j2, xi1j0, xi2j0, difference=0):
    center = img[15][15]
    others = [xi0j1, xi0j2, xi1j0, xi2j0]
    booleans = [xi0j0 != 0
                , all(x != 0 for x in others)
                , all(x < xi0j0 for x in others)
                , all(x >= ((1-difference)*center) for x in others)
               ]
    return all(booleans)

# boolean function: determines if there's a peak within a given distance
#   around center point (target star)
def has_close_peaks(target, diff=7):
    img = target.img
    len_x = len(img[0])
    len_y = len(img)
    c_i = len_x%2
    c_j = len_y%2
    main_peak = target.img[c_i][c_j]
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
            if is_peak(img, img[i][j], img[i][j-1], img[i][j+1], img[i-1][j], img[i+1][j]):
                logger.info("has_close_peaks True")
                return True
    logger.info("has_close_peaks False")
    return False

# boolean function: determines if aperture has more than one bright peak
def has_peaks(target):
    main_peak = target.img[15][15]
    img = target.img
#    peaks = 1
#    peak_locations = []
    for i in range(2, len(img) - 2):
        for j in range(2, len(img[i]) - 2):
            if i-1 == 14 or j-1 == 14 or i+1 == 16 or j+1 == 16:
                continue
            if is_peak(img, img[i][j], img[i][j-1], img[i][j+1], img[i-1][j], img[i+1][j]):
                logger.info("has_peaks True")
                return True
#               peaks += 1
#               peak_locations.append((i, j))
    logger.info("has_peaks False")
    return False

# get list of kics from a file where kic is first of a column
def get_kics(filename):
    all_kics = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            all_kics.append(row[0])
    logger.info("get_kics done")
    return all_kics

# sets up photometry for a star and adds aperture to class
def run_photometry(targ, image_region=15, edge_lim=0.015, min_val=5000, \
                   ntargets=100, extend_region_size=3, remove_excess=4):
    target = photometry.star(targ)
    target.make_postcard()

    jj, ii = target.center
    jj, ii = int(jj), int(ii)

    if ii - image_region <= 0 or jj - image_region <=0:
        logger.info("run_photometry unsuccessful")
        return 1

    target.find_other_sources(edge_lim, min_val, ntargets, extend_region_size, \
                              remove_excess, plot_flag=False)
    target.data_for_target(do_roll=True, ignore_bright=0)

    img = np.sum(((target.targets == 1)*target.postcard + (target.targets == 1)*100000)
                     [:,jj-image_region:jj+image_region,ii-image_region:ii+image_region], axis=0)
    setattr(photometry.star, 'img', img)

    logger.info("run_photometry done")
    return target

# helper function for plot_targets that sets up photometry of a star
#   runs a photometry and tests a list of boolean functions on it
#   then creates a plot for it with plot_data
def tests_booleans(targ, boolean_funcs, count, image_region=15, edge_lim=0.015, min_val=500):
    target = run_photometry(targ, image_region, edge_lim, min_val)
    if target == 1:
        return 1
    for boolean in boolean_funcs:
        if boolean(target):
            return 1
    logger.info("tests_booleans done")
    return plot_data(target, count, image_region)

# outputs dict of functions that finds faulty stars
#   and kics that fall in those functions
def get_boolean_stars(targets, boolean_funcs, image_region=15, edge_lim=0.015, min_val=500):
    full_dict = {}
    full_dict["good"] = []
    for boolean_func in boolean_funcs:
        full_dict[boolean_func.__name__] = []
    for targ in targets:
        is_faulty = False
        target = run_photometry(targ, image_region, edge_lim, min_val)
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
    total = len(targets)
    count = 1
    parsed_targets = []
    if len(boolean_funcs) == 1:
        output_file = filename + str(boolean_funcs[0].__name__) + ".pdf"
    else:
        output_file = filename + "_bads.pdf"
    with PdfPages(output_file) as pdf:
        for target in targets:
            logger.info("current: " + target)
            if tests_booleans(target, boolean_funcs, count) == 0:
                parsed_targets.append(target)
                pdf.savefig()
                plt.close()
                logger.info(str(count) + "\t" + target + "\tplot_done")
                count += 1
    logger.info(str(count - 1) + " out of " + str(total) + " targets plotted")
    logger.info("plot_targets done")
    return parsed_targets

def is_std_better_biggest(old_stds, stds):
    max_i = np.argmax(stds)
    return stds[max_i] <= old_stds[max_i]

def is_std_better_avg(old_stds, stds):
    return np.average(stds) <= np.average(old_stds)

def run_partial_photometry(target, image_region=15, edge_lim=0.015, min_val=5000, \
                           ntargets=100, extend_region_size=3, remove_excess=4):
    target.find_other_sources(edge_lim=edge_lim, min_val=min_val, ntargets=ntargets, \
                              extend_region_size=extend_region_size, plot_flag=False)
    target.data_for_target(do_roll=True, ignore_bright=0)
    jj, ii = target.center
    jj, ii = int(jj), int(ii)
    img = np.sum(((target.targets == 1)*target.postcard + (target.targets == 1)*100000)
                 [:,jj-image_region:jj+image_region,ii-image_region:ii+image_region], axis=0)
    setattr(photometry.star, 'img', img)
    logger.info("run_partial_photometry done")
    return 0

# TODO: gets better apertures
def print_better_apertures(targ, boolean_func, edge_lim=0.015, min_val=5000,
                           extend_region_size=3, remove_excess=4, image_region=15):
    count = 0

    target = photometry.star(targ)
    target.make_postcard()

    edge_lims = np.arange(edge_lim - 0.010, edge_lim + 0.025, 0.005)
    min_vals = np.arange(min_val - 2000, min_val + 2000, 500)
    region_sizes = np.arange(2, 5)
    excesses = np.arange(2, 6)

    vals = []
    for val_1 in edge_lims:
        for val_2 in min_vals:
            vals.append((val_1, val_2))

    results = {}

    run_partial_photometry(target, edge_lim=0.015, min_val=5000, \
                           extend_region_size=3, remove_excess=4, ntargets=100)
    old_stds = target.flux_uncert
    plot_data(target, count, image_region)

    with PdfPages(output_file) as pdf:
        for val in vals:
            count += 1
            res = {}
            run_partial_photometry(target, edge_lim=val[0], min_val=val[1], \
                                   extend_region_size=extend_region_size, \
                                   remove_excess=remove_excess, ntargets=100)
            res["settings"] = val
            res["boolean"] = boolean_func(target)
            res["is_avg"] = is_std_better_avg(old_stds, target.flux_uncert)
            res["is_most"] = is_std_better_biggest(old_stds, target.flux_uncert)
            res["has_peaks"] = has_close_peaks(target)
            results[val] = res
            plot_data(target, count, image_region)
            plt.gcf().text(4/8.5, 1/11., str(res), ha='center', fontsize = 12)
            pdf.savefig()
            plt.close()
    logger.info("get_better_apertures done")
    return

def testing():
    targ = "893033"
    target = photometry.star(targ)
    target.make_postcard()

    target.find_other_sources(edge_lim=0.08, min_val=5000, ntargets=100, \
                              extend_region_size=3, remove_excess=4, plot_flag=False)
    target.data_for_target(do_roll=True, ignore_bright=0)

    print(is_large_ap(target))
    plot_data(target, count=0, image_region=15)
    plt.show()
    logger.info("testing done")
    return

def fake_bool(target):
    logger.info("fake_bool done")
    return False

def main():
    logger.info("### starting ###")
    # gets good kics to be put into MAST
    # array_to_file(get_good_kids(stellar_param_filename, get_kids(list_filenames)))

    # uses MAST results to get stars without neighbours
    # remove_bright_neighbours_separate()

    # plots list of files with kics using f3
    # plot_targets(output_file, [fake_bool], ["893033"])
    # plot_targets(targets_file, [is_large_ap, has_peaks], get_kics(targets_file))

    # testing()
    print_better_apertures("893033", is_large_ap)
    logger.info("### everything done ###")

if __name__ == "__main__" and __package__ is None:
    from os import sys, path
    sys.path.append(f3_location)
    from f3 import photometry
    main()
