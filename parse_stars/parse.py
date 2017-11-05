import csv
import os

## Variables
#filename_periods = "sample_periodic.txt"
#filename_nonperiods = "sample_nonperiodic.txt"
filename_periods = "Table_Periodic.txt"
filename_nonperiods = "Table_Non_Periodic.txt"
list_filenames = [filename_periods, filename_nonperiods]

#stellar_param = "sample_table.txt"
stellar_param = "table4.dat"
stellar_params_i = [1, 4, 7] #KIC = 0, Teff, logg, Fe/H

#good_kids_filename = "good_kids.txt"
good_kids_file = "good_kids"

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
    print("get_kids done")
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
    print("get_good_kids done")
    return list_good_kids

# prints array to column in file, splits up files by number_kids_per_file
def array_to_file(arr):
    arr_i = 0
    total_kids = len(arr)
    number_kids_per_file = 2
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

    print("array_to_file done")
    return 0

#
def remove_bright_neighbours(list_kids, kids_kepmag):
    difference_max = 2.0
    return 0

## Tests
#array_to_file(["test","1","thurd", "fourrrr", "fiffff"])
array_to_file(get_good_kids(stellar_param, get_kids(list_filenames)))
