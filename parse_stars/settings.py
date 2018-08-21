"""
GLOBAL VARIABLES AND LOCATIONS AND CODE LOGGING TO SETUP CODE
"""

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
filename_mast_table = "./data/kepler_fov_search_all.txt"
kepmag_file_prefix = "./data/kepler_fov_search"
filename_lc_img = "./results/lc_data_img.out"
filename_lc_new = "./results/lc_data_new.out"
filename_lc_old = "./results/lc_data_old.out"
filename_lc_obs = "./data/obs_info.txt"

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


## Headings/types for databases to use

# Mathur
updated_dic_keys = ["KIC", "Teff", "E_Teff", "e_Teff", "log(g)", "E_log(g)", "e_log(g)", \
                    "[Fe/H]", "E_[Fe/H]", "e_[Fe/H]", "Rad", "E_Rad", "e_rad", \
                    "Mass", "E_Mass", "e_Mass", "rho", "E_rho", "e_rho",
                    "Dist", "E_Dist", "e_Dist", "Av", "E_Av", "e_Av", "Mod"]
updated_dic = {
    "kic": ("KIC", int),
    "teff": ("Teff", int), # Effective temperature; output
    "E_Teff": ("E_Teff", int), # Upper 1{sigma} confidence interval
    "e_Teff": ("e_Teff", int), # Lower 1{sigma} confidence interval
    "logg": ("log(g)", float), # Log surface gravity; output
    "E_log(g)": ("E_log(g)", float), # Upper 1{sigma} confidence interval
    "e_log(g)": ("e_log(g)", float), # Lower 1{sigma} confidence interval
    "metallicity": ("[Fe/H]", float), # Metallicity; output
    "E_[Fe/H]": ("E_[Fe/H]", float), # Upper 1{sigma} confidence interval
    "e_[Fe/H]": ("e_[Fe/H]", float), # Lower 1{sigma} confidence interval
    "rad": ("Rad", float), # Radius
    "E_Rad": ("E_Rad", float), # Upper 1{sigma} confidence interval
    "e_rad": ("e_rad", float), # Lower 1{sigma} confidence interval
    "mass": ("Mass", float), # Mass
    "E_Mass": ("E_Mass", float), # Upper 1{sigma} confidence interval
    "e_Mass": ("e_Mass", float), # Lower 1{sigma} confidence interval
    "rho": ("rho", float), # Density
    "E_rho": ("E_rho", float), # Upper 1{sigma} confidence interval
    "e_rho": ("e_rho", float), # Lower 1{sigma} confidence interval
    "dist": ("Dist", float), # Distance
    "E_Dist": ("E_Dist", float), # Upper 1{sigma} confidence interval
    "e_Dist": ("e_Dist", float), # Lower 1{sigma} confidence interval
    "av": ("Av", float), # V band extinction
    "E_Av": ("E_Av", float), # Upper 1{sigma} confidence interval
    "e_Av": ("e_Av", float), # Lower 1{sigma} confidence interval
    "mod": ("Mod", str) # Provenance of output values
}

# McQuillian
periodic_dic_keys = ["KID", "Teff", "logg", "Mass", "Prot", "Prot_err", "Rper", \
                     "LPH", "w", "DC", "Flag"]
periodic_dic = {
    "KID": ("KID", int),
    "Teff": ("Teff", int), # K, Dressing & Charbonneau
    "logg": ("logg", float),
    "Mass": ("Mass", float),
    "prot": ("Prot", float), # Period, days
    "Prot_err": ("Prot_err", float), #Period error
    "rper": ("Rper", float), #Average amplitude of variability within one period, ppm
    "LPH": ("LPH", float), # Local peak height
    "w": ("w", float), # Assigned weight
    "DC": ("DC", int),
    "Flag": ("Flag", str) # Corrections/smoothing
}

# McQuillian
nonperiodic_dic_keys = ["KID", "Teff", "logg", "Mass", "Prot", "Prot_err", "LPH", "w", "DC"]
nonperiodic_dic = {
    "KID": ("KID", int),
    "Teff": ("Teff", int),
    "logg": ("logg", float),
    "Mass": ("Mass", float),
    "prot": ("Prot", float),
    "Prot_err": ("Prot_err", float),
    "LPH": ("LPH", float),
    "w": ("w", float),
    "DC": ("DC", int),
}

# MAST database/kplr
mast_params = ["kic_altsource"
               "kic_scpkey"
               "kic_tmid"
               "kic_dec"
               "kic_glon"
               "kic_kepmag"
               "kic_hmag"
               "kic_gmag"
               "kic_degree_ra"
               "kic_kepler_id"
               "kic_altid"
               "kic_pmdec"
               "kic_feh"
               "kic_catkey"
               "kic_kmag"
               "kic_jmag"
               "kic_galaxy"
               "kic_av"
               "kic_pmtotal"
               "kic_glat"
               "kic_ebminusv"
               "kic_teff"
               "kic_gkcolor"
               "kic_rmag"
               "kic_imag"
               "kic_2mass_id"
               "kic_zmag"
               "kic_grcolor"
               "flag"
               "angular_separation"
               "kic_aq"
               "kic_radius"
               "kic_logg"
               "kic_blend"
               "kic_d51mag"
               "kic_gredmag"
               "kic_variable"
               "kic_jkcolor"
               "kic_cq"
               "kic_ra"
               "kic_parallax"
               "kic_pmra"
               "kic_pq"
               "kic_umag"
               "kic_scpid"
               "U_UBV"
               "masterRA"
               "masterDec"
               "gr"
               "Channel_1"
               "Channel_0"
               "Channel_3"
               "Channel_2"
               "kic_kepler_id"
               "Module_1"
               "Module_0"
               "Module_3"
               "Module_2"
               "angular_separation"
               "twomass_2mass_id"
               "Column_3"
               "Column_2"
               "Column_1"
               "Column_0"
               "kic_parallax"
               "Row_0"
               "Row_1"
               "Row_2"
               "Row_3"
]


