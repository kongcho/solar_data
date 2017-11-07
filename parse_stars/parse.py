import csv
import os
import logging
import re

## Input Files to change
filename_periods = "sample_periodic.txt"
filename_nonperiods = "sample_nonperiodic.txt"
#filename_periods = "Table_Periodic.txt"
#filename_nonperiods = "Table_Non_Periodic.txt"
list_filenames = [filename_periods, filename_nonperiods]

stellar_param_filename = "sample_table.txt"
#stellar_param_filename = "table4.dat"
stellar_params_i = [1, 4, 7] #KIC = 0, Teff, logg, Fe/H

good_kids_file = "good_kids"

kepmag_file_prefix = "kepler_fov_search"

## Logging
log_file = good_kids_file + ".log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_file)
handler.setLevel(logging.INFO)

## Functions
# assume all kids are unique from each file
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
# assume all kids are unique
def get_good_kids(param_filename, list_kids):
    list_good_kids = []
    with open(param_filename) as f_param:
        for line in f_param:
            curr_param_kid = line.strip().split(' ')[0]
            for kids_i in range(len(list_kids)):
                if curr_param_kid == list_kids[kids_i]:
                    list_good_kids.append(curr_param_kid)
    logger.info("get_good_kids done")
    return list_good_kids

# prints array to column in file, splits up files by number_kids_per_file
# returns length of array
def array_to_file(arr):
    arr_i = 0
    total_kids = len(arr)
    number_kids_per_file = 20000
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
    logger.info(str(total_kids) + " entries in " + str(number_of_files) + " files")
    logger.info("array_to_file done")
    return total_kids

# assumes that list of neighbour stars follows from target stars
# assumes list of input stars has same order and is same as processed stars
def remove_bright_neighbours():
    kepmag_col_no = 1
    difference_max = 2.0
    curr_id = -1
    count = 0
    parse_kids_filename = good_kids_file + "_parsed.txt"

    if os.path.exists(parse_kids_filename):
        ans = input("File already exists, do you want to proceed for all? (y/n) ")
        ans = ans.strip().lower()
        if (ans != "y"):
            print("Quitting...")
            return 1

    input_files = sorted([filename for filename in os.listdir('.') if filename.startswith(kepmag_file_prefix)])

    with open(parse_kids_filename, 'w') as output_f:
        for input_files_i in range(len(input_files)):
            with open(input_files[input_files_i]) as input_f:
                for line in input_f:
                    curr_line = line.strip()
                    if curr_line[0:10] == "Input line": #id line
                        curr_id = curr_line[14:]
                        output_f.write("id " + str(count) + ": " + curr_id + "\n")
                        output_f.write(curr_id + "\n")
                        count += 1
                        if curr_line[10:12] == " 1": # line under input 1 is labels
                            fieldnames = input_f.readline().strip().split(',')
                            for fields_i in range(len(fieldnames)):
                                if fieldnames[fields_i] == "Kepler_ID":
                                    kid_col_no = fields_i
                                if fieldnames[fields_i] == "kepmag":
                                    kepmag_col_no = fields_i
                            input_f.readline() #types
                        curr_data = input_f.readline().strip().split(',')
                        curr_kepmag = float(curr_data[kepmag_col_no])
                        if curr_id != curr_data[kid_col_no]:
                            print("current kid is not correct") #shouldn't happen if format is correct
                    else:
                        test_data = curr_line.split(',')
                        if abs(float(test_data[kepmag_col_no]) - curr_kepmag) <= 2.0:
                            output_f.write(str(test_data[kid_col_no]) + "\n")
    logger.info("printed " + str(count) + " kids")
    logger.info("remove_bright_neighbours done")
    return 0

## Tests
#array_to_file(get_good_kids(stellar_param_filename, get_kids(list_filenames)))
remove_bright_neighbours()
