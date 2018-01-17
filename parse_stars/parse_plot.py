import csv
import os
import logging
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.backends.backend_pdf import PdfPages

# Setup
f3_location = "/home/user/Desktop/astrolab/solar_data/f3"

file_name = "sample_single"
targets_files = [file_name]

targets_files_temp = ["./data/" + x.split('.')[0]
                      for x in os.listdir("./data/")
                      if x.endswith(".txt") and "single" in x]
if len(targets_files) != 0:
    file_name = targets_files_temp[0]
    targets_files = targets_files_temp

targets_file = file_name + ".txt"
output_file = file_name + "_plot.pdf"
log_file = file_name + ".log"

## Input Files to change
filename_periods = "./data/Table_Periodic.txt"
filename_nonperiods = "./data/Table_Non_Periodic.txt"
list_filenames = [filename_periods]

stellar_param_filename = "./data/table4.dat" #KIC = 0, Teff, logg, Fe/H

good_kids_file = "results/good_kids"

kepmag_file_prefix = "./data/kepler_fov_search"

## Logging
log_file = good_kids_file + ".log"
logger_main = logging.getLogger(__name__)
logging.basicConfig(filename=log_file, level=logging.INFO)

logger = logging.getLogger(__name__)
handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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
    logging.info("get_kids done")
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
    logging.info("get_existing_kids done")
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
    logging.info("get_good_kids done")
    return list_good_kids

# prints array to column in file, splits up files by number_kids_per_file
# returns length of array
def array_to_file(arr):
    arr_i = 0
    total_kids = len(arr)
    number_kids_per_file = 9999
    output_files_i = 0
    number_of_files = total_kids//number_kids_per_file + 1
    remainder_of_files = total_kids%number_kids_per_file
    good_kids_filename = good_kids_file + "_" + str(output_files_i) + ".txt"

    if os.path.exists(good_kids_filename):
        ans = input("File already exists, do you want to proceed for all? (y/n) ")
        ans = ans.strip().lower()
        if (ans != "y"):
            print("Quitting...")
            return 1

    for output_files_i in range(number_of_files):
        good_kids_filename = good_kids_file + "_" + str(output_files_i) + ".txt"
        if (output_files_i == number_of_files - 1):
            number_kids_per_file = remainder_of_files
        with open(good_kids_filename, 'w') as write_f:
            writer = csv.writer(write_f, delimiter=',', lineterminator='\n')
            for i in range(arr_i, arr_i + number_kids_per_file):
                writer.writerow([arr[i]])
            arr_i = arr_i + number_kids_per_file
    logging.info(str(total_kids) + " entries in " + str(number_of_files) + " files")
    logging.info("array_to_file done")
    return total_kids

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
def remove_bright_neighbours_together():
    all_kids = []
    kepmag_col_no = 1
    difference_max = 2.0
    curr_id = -1
    count = 0
    parsed_kids_filename = good_kids_file + "_parsed.txt"

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
    logging.info("printed " + str(count) + " kids")
    logging.info("remove_bright_neighbours_together done")
    return 0

# doesn't remove duplicates
# outputs separate files for stars with no neighbours and with neighbours
def remove_bright_neighbours_separate():
    difference_max = 2.0

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
    single_parsed_kids_filename = good_kids_file + "_single.txt"
    batch_parsed_kids_filename = good_kids_file + "_batch.txt"

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
    logging.info("printed " + str(len(single_kids)) + " kids with no neighbours")
    logging.info("printed " + str(len(batch_kids)) + " kids with neighbours")
    logging.info("remove_bright_neighbours_separate done")
    return 0

def plot_data(targ, do_roll=True, ignore_bright=0, image_region=15):

    targ.data_for_target(do_roll, ignore_bright)

    fig = plt.figure(figsize=(11,8))
    gs.GridSpec(3,3)

    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    for i in range(4):
        g = np.where(targ.qs == i)[0]
        plt.errorbar(targ.times[g], targ.obs_flux[g], yerr=targ.flux_uncert[i], fmt=targ.fmt[i])

    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Relative Flux', fontsize=15)

    plt.subplot2grid((3,3), (1,2))
    plt.title(targ.kic, fontsize=20)
    jj, ii = targ.center
    jj, ii = int(jj), int(ii)
    img = np.sum(((targ.targets == 1)*targ.postcard + (targ.targets == 1)*100000)
                     [:,jj-image_region:jj+image_region,ii-image_region:ii+image_region], axis=0)
    plt.imshow(img, interpolation='nearest', cmap='gray', vmin=98000*52, vmax=104000*52)

    fig.text(7.5/8.5, 0.5/11., str(count + 1), ha='center', fontsize = 12)
    fig.tight_layout()

def run_photometry(targ):
    target = photometry.star(targ)

    target.make_postcard()

    target.find_other_sources(ntargets=100, plot_flag=False)

    plot_data(target)

def plot_targets(files):
    with PdfPages(output_file) as pdf:
        for single_file in files:
            targets_file = single_file + ".txt"
            with open(targets_file) as f:
                reader = csv.reader(f)
                for count, row in enumerate(reader):
                    target = row[0]
                    run_photometry(target)
                    pdf.savefig()
                    plt.close()
                    logger.info(str(count + 1) + "\t" + target + "\tplot_done")
    logger.info("plot_all_targets_done")

def is_more_or_less(target, quarter):
    for indexes in quarter:
        is_incr = False
        is_decr = False
        for i in range(len(indexes - 2)):
            if target.obs_flux[indexes[i + 1]] > target.obs_flux[indexes[i]]:
                if is_decr:
                    break
                is_incr = True
            else if target.obs_flux[indexes[i + 1]] < target.obs_flux[indexes[i]]:
                if is_incr:
                    break
                is_decr = True
        return 1 if is_incr else -1 if is_decr else 0

def is_large_ap(target):
    golden = range(0, 8)
    blacks = [[8, 9], [20, 21, 22], [31, 32, 33], [43, 44, 45]]
    reds = [[10, 11, 12], [23, 24, 25], [34, 35, 36], [46, 47, 48]]
    blues = [[13, 14, 15, 16], [26, 27], [37, 38, 39], [49, 50, 51]]
    greens = [[17, 18, 19], [28, 29, 30], [40, 41, 42]]
    channels = [blacks, reds, blues, greens]
    for channel in channels:
        curr_ch = []
        is_strange = False
        for qtr_i in range(len(channel) - 2):
            if channel[qtr_i + 1] is not channel[qtr]:
                break
            else:
                is_strange = True
    return is_strange


def categorise(target):
    categories = ["large_ap", "small_ap", "normal"]
    return is_large_ap(target)

## Main
#array_to_file(get_good_kids(stellar_param_filename, get_kids(list_filenames)))
#remove_bright_neighbours_separate()

def main():
    plot_targets(targets_files)

if __name__ == "__main__" and __package__ is None:
    from os import sys, path
    sys.path.append(f3_location)
    from f3 import photometry
    main()
