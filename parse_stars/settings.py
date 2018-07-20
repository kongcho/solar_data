import logging
import sys
import os

## Setup
f3_location = "./f3"
results_folder = "./results/"
output_folder = "./tests/"
ffidata_folder = "./ffidata/"


## Input Files to change
filename_periods = "./data/Table_Periodic.txt"
filename_nonperiods = "./data/Table_Non_Periodic.txt"
filename_stellar_params = "./data/table4.dat" #KIC = 0, Teff, logg, Fe/H
kepmag_file_prefix = "./data/kepler_fov_search"


## Output Files to change
file_name = "good_kids"

## Output Files
log_file = output_folder + file_name + ".log"
output_file = output_folder + file_name + "_plot.pdf"
parsed_kids_filename = results_folder + file_name + "_parsed.txt"
single_parsed_kids_filename = results_folder + file_name + "_single.txt"
batch_parsed_kids_filename = results_folder + file_name + "_batch.txt"

targets_file = single_parsed_kids_filename

# picks up list of txt files with "single" in the name
def set_filenames_based_on_folders(data_folder="./data/"):
    targets_files_temp = [data_folder + x.split('.')[0] + ".txt"
                          for x in os.listdir(data_folder)
                          if x.endswith(".txt") and "single" in x]
    if len(targets_files) != 0:
        file_name = targets_files_temp[0]
        targets_files = targets_files_temp


## Matplotlib Setup
def mpl_setup():
    import matplotlib
    if os.environ.get('DISPLAY','') == "":
        print("Using non-interactive Agg backend")
        matplotlib.use('Agg')


## Logging
# removes warning messages from kplr that is propagated to f3
# setup logging for different modules
def setup_logging():
    format_str = '%(asctime)s [%(levelname)s]\t%(name)s-%(module)-10s - %(funcName)-20s: %(message)s'
    formatter = logging.Formatter(format_str)

    # for stderr
    logging.basicConfig(stream=sys.stderr, level=logging.ERROR, format=format_str)
    root = logging.getLogger()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    return logger

# setup logging for when file is being run and not imported
def setup_main_logging():
    format_str = '%(asctime)s [%(levelname)s]\t%(name)s-%(module)-10s - %(funcName)-20s: %(message)s'
    formatter = logging.Formatter(format_str)

    # for stderr
    logging.basicConfig(stream=sys.stderr, level=logging.ERROR, format=format_str)
    root = logging.getLogger()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # for the file
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
