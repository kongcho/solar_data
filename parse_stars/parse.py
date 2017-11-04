import csv

# Variables
filename_periods = "sample_periodic.txt"
filename_nonperiods = "sample_nonperiodic.txt"
list_filenames = [filename_periods, filename_nonperiods]

stellar_param = "sample_table.txt"
stellar_params_i = [1, 4, 7] #KIC = 0, Teff, logg, Fe/H

# Functions

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
    return list_good_kids

# Tests
#print(get_kids(list_filenames))
print(get_good_kids(stellar_param, get_kids(list_filenames)))
