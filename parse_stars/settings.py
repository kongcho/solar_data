import logging
import sys
import os

## Setup
f3_location = "./f3"
results_folder = "./results/"
output_folder = "./tests/"
ffidata_folder = "./ffidata/"
data_folder = "./data/"

## Input Files to change
filename_periods = data_folder + "Table_Periodic.txt"
filename_nonperiods = data_folder + "Table_Non_Periodic.txt"
filename_stellar_params = data_folder + "table4.dat"
filename_mast_table = data_folder + "kepler_fov_search_all.txt"
filename_kic10_table = data_folder + "kic.txt"
filename_gaia_table = data_folder + "DR2PapTable1.txt"
kepmag_file_prefix = data_folder + "kepler_fov_search"

filename_lc_img = results_folder + "lc_data_img.out"
filename_lc_new = results_folder + "lc_data_new.out"
filename_lc_old = results_folder + "lc_data_old.out"
filename_lc_var = results_folder + "trf_var_out_FINAL.out"
filename_lc_obs = data_folder + "obs_info.txt"

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
    #%(name)s
    format_str = '%(asctime)s [%(levelname)s]\t%(module)-6s - %(funcName)-20s: %(message)s'
    formatter = logging.Formatter(format_str)

    # for stderr
    logging.basicConfig(stream=sys.stderr, level=logging.ERROR, format=format_str)
    root = logging.getLogger()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    return logger

# setup logging for when file is being run and not imported
def setup_main_logging():
    format_str = '%(asctime)s [%(levelname)s]\t%(module)-6s - %(funcName)-20s: %(message)s'
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

gaia_dic_keys = ["KIC", "source_id", "teff", "teffe", "dis", "disep", "disem", \
                 "rad", "radep", "radem", "avs", "evState", "binaryFlag"]
gaia_dic = {
    "KIC": ("KIC", int),
    "source_id": ("source_id", int),
    "teff": ("teff", int),
    "teffe": ("teffe", int),
    "dist": ("dis", float),
    "disep": ("disep", float),
    "disem": ("disem", float),
    "rad": ("rad", float),
    "radep": ("radep", float),
    "radem": ("radem", float),
    "avs": ("avs", float),
    "evState": ("evState", int), # evolutionary state
    "binaryFlag": ("binaryFlag", int) # binarity
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

kic10_dic_keys = ["kic_ra", "kic_dec", "kic_pmra", "kic_pmdec", "kic_umag", "kic_gmag", "kic_rmag", "kic_imag", "kic_zmag", "kic_gredmag", "kic_d51mag", "kic_jmag", "kic_hmag", "kic_kmag", "kic_kepmag", "kic_kepler_id", "kic_tmid", "kic_scpid", "kic_altid", "kic_altsource", "kic_galaxy", "kic_blend", "kic_variable", "kic_teff", "kic_logg", "kic_feh", "kic_ebminusv", "kic_av", "kic_radius", "kic_cq", "kic_pq", "kic_aq", "kic_catkey", "kic_scpkey", "kic_parallax", "kic_glon", "kic_glat", "kic_pmtotal", "kic_grcolor", "kic_jkcolor", "kic_gkcolor", "kic_degree_ra", "kic_fov_flag", "kic_tm_designation"]

kic10_dic = {
    "ra": ("kic_ra", float),
    "dec": ("kic_dec", float),
    "pmra": ("kic_pmra", float),
    "pmdec": ("kic_pmdec", float),
    "umag": ("kic_umag", float),
    "gmag": ("kic_gmag", float),
    "rmag": ("kic_rmag", float),
    "imag": ("kic_imag", float),
    "zmag": ("kic_zmag", float),
    "gredmag": ("kic_gredmag", float),
    "d51mag": ("kic_d51mag", float),
    "jmag": ("kic_jmag", float),
    "hmag": ("kic_hmag", float),
    "kmag": ("kic_kmag", float),
    "kepmag": ("kic_kepmag", float),
    "kic": ("kic_kepler_id", int),
    "tmid": ("kic_tmid", int),
    "scpid": ("kic_scpid", int),
    "altid": ("kic_altid", int),
    "altsource": ("kic_altsource", int),
    "galaxy": ("kic_galaxy", int),
    "blend": ("kic_blend", int),
    "variable": ("kic_variable", int),
    "teff": ("kic_teff", int),
    "logg": ("kic_logg", float),
    "feh": ("kic_feh", float),
    "ebminusv": ("kic_ebminusv", float),
    "av": ("kic_av", float),
    "rad": ("kic_radius", float),
    "cq": ("kic_cq", str),
    "pq": ("kic_pq", int),
    "aq": ("kic_aq", int),
    "catkey": ("kic_catkey", int),
    "scpkey": ("kic_scpkey", int),
    "parallax": ("kic_parallax", float),
    "glon": ("kic_glon", float),
    "glat": ("kic_glat", float),
    "pmtotal": ("kic_pmtotal", float),
    "grcolor": ("kic_grcolor", float),
    "jkcolor": ("kic_jkcolor", float),
    "gkcolor": ("kic_gkcolor", float),
    "degree_ra": ("kic_degree_ra", float),
    "fov_flag": ("kic_fov_flag", int),
    "tm_designation": ("kic_tm_designation", int)
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


