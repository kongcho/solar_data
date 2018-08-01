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


updated_dic = {
    "kic": ("KIC", int),
    "Teff": ("Teff", int), # Effective temperature; output
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

mast_kic10_dic = {
    "kic_2mass_id": ("kic_2mass_id", str),
    "kic_hmag": ("kic_hmag", float),
    "kic_jmag": ("kic_jmag", float),
    "kic_kmag": ("kic_kmag", float),
    "kic_aq": ("kic_aq", int),
    "kic_variable": ("kic_variable", int),
    "kic_d51mag": ("kic_d51mag", float),
    "kic_pmdec": ("kic_pmdec", float),
    "kic_dec": ("kic_dec", float),
    "kic_av": ("kic_av", float),
    "kic_teff": ("kic_teff", int),
    "kic_ebminusv": ("kic_ebminusv", float),
    "kic_feh": ("kic_feh", float),
    "kic_logg": ("kic_logg", float),
    "kic_radius": ("kic_radius", float),
    "kic_gmag": ("kic_gmag", float),
    "kic_gkcolor": ("kic_gkcolor", float),
    "kic_grcolor": ("kic_grcolor", float),
    "kic_glat": ("kic_glat", float),
    "kic_glon": ("kic_glon", float),
    "kic_gredmag": ("kic_gredmag", float),
    "kic_imag": ("kic_imag", float),
    "kic_altid": ("kic_altid", int),
    "kic_blend": ("kic_blend", int),
    "kic_jkcolor": ("kic_jkcolor", float),
    "kic_kepmag": ("kic_kepmag", float),
    "kic_parallax": ("kic_parallax", float),
    "kic_pq": ("kic_pq", int),
    "kic_rmag": ("kic_rmag", float),
    "kic_pmra": ("kic_pmra", float),
    "kic_degree_ra": ("kic_degree_ra", float),
    "kic_ra": ("kic_ra", float),
    "kic_altsource": ("kic_altsource", int),
    "kic_cq": ("kic_cq", str),
    "kic_galaxy": ("kic_galaxy", int),
    "kic_pmtotal": ("kic_pmtotal", float),
    "kic_umag": ("kic_umag", float),
    "kic_tmid": ("kic_tmid", int),
    "kic_catkey": ("kic_catkey", int),
    "kic_scpkey": ("kic_scpkey", int),
    "kic": ("kic_kepler_id", int),
    "kic_scpid": ("kic_scpid", int),
    "kic_zmag": ("kic_zmag", float),
}

mast_fov_dic = {
    "masterRA": ("masterRA", float),
	"masterDec": ("masterDec", float),
	"masterID": ("masterID", str),
	"kic_kepler_id": ("kic_kepler_id", int),
	"twomass_2mass_id": ("twomass_2mass_id", str),
	"twomass_tmid": ("twomass_tmid", int),
	"twomass_conflictflag": ("twomass_conflictflag", str),
	"kic_degree_ra": ("kic_degree_ra", float),
	"kic_dec": ("kic_dec", float),
	"kct_avail_flag": ("kct_avail_flag", int),
	"kct_num_season_onccd": ("kct_num_season_onccd", int),
	"kic_pmra": ("kic_pmra", float),
	"kic_pmdec": ("kic_pmdec", float),
	"kic_scpid": ("kic_scpid", int),
	"kic_altid": ("kic_altid", int),
	"kic_altsource": ("kic_altsource", int),
	"kic_galaxy": ("kic_galaxy", int),
	"kic_variable": ("kic_variable", int),
	"kic_teff": ("kic_teff", int),
	"kic_logg": ("kic_logg", float),
	"kic_feh": ("kic_feh", float),
	"kic_ebminusv": ("kic_ebminusv", float),
	"kic_av": ("kic_av", float),
	"kic_radius": ("kic_radius", float),
	"kic_cq": ("kic_cq", str),
	"kic_pq": ("kic_pq", int),
	"kic_aq": ("kic_aq", int),
	"kic_catkey": ("kic_catkey", int),
	"kic_scpkey": ("kic_scpkey", int),
	"kic_parallax": ("kic_parallax", float),
	"glon": ("glon", float),
	"glat": ("glat", float),
	"kic_pmtotal": ("kic_pmtotal", float),
	"kic_ra": ("kic_ra", float),
	"g": ("g", float),
	"r": ("r", float),
	"i": ("i", float),
	"z": ("z", float),
	"gred": ("gred", float),
	"d51mag": ("d51mag", float),
	"J": ("J", float),
	"H": ("H", float),
	"K": ("K", float),
	"kepmag": ("kepmag", float),
	"FUV": ("FUV", float),
	"NUV": ("NUV", float),
	"ID_USNO": ("ID_USNO", str),
	"ra_USNO": ("ra_USNO", float),
	"dec_USNO": ("dec_USNO", float),
	"ID_GALEX": ("ID_GALEX", str),
	"ra_GALEX": ("ra_GALEX", float),
	"dec_GALEX": ("dec_GALEX", float),
	"ra_2M": ("ra_2M", float),
	"dec_2M": ("dec_2M", float),
	"ID_IRT": ("ID_IRT", str),
	"ra_IRT": ("ra_IRT", float),
	"dec_IRT": ("dec_IRT", float),
	"pStar_IRT": ("pStar_IRT", float),
	"J_IRT": ("J_IRT", float),
	"JErr_IRT": ("JErr_IRT", float),
	"jClass_IRT": ("jClass_IRT", int),
	"jppErrBits_IRT": ("jppErrBits_IRT", int),
	"Sep_IRT": ("Sep_IRT", float),
	"gr": ("gr", float),
	"gi": ("gi", float),
	"iz": ("iz", float),
	"gK": ("gK", float),
	"JH": ("JH", float),
	"HK": ("HK", float),
	"JK": ("JK", float),
	"gJ": ("gJ", float),
	"FUVNUV": ("FUVNUV", float),
	"NUVg": ("NUVg", float),
	"gJ_KIC_IRT": ("gJ_KIC_IRT", float),
	"kct_sky_group_id": ("kct_sky_group_id", int),
	"kct_crowding_season_0": ("kct_crowding_season_0", float),
	"kct_crowding_season_1": ("kct_crowding_season_1", float),
	"kct_crowding_season_2": ("kct_crowding_season_2", float),
	"kct_crowding_season_3": ("kct_crowding_season_3", float),
	"kct_contamination_season_0": ("kct_contamination_season_0", float),
	"kct_contamination_season_1": ("kct_contamination_season_1", float),
	"kct_contamination_season_2": ("kct_contamination_season_2", float),
	"kct_contamination_season_3": ("kct_contamination_season_3", float),
	"kct_flux_fraction_season_0": ("kct_flux_fraction_season_0", float),
	"kct_flux_fraction_season_1": ("kct_flux_fraction_season_1", float),
	"kct_flux_fraction_season_2": ("kct_flux_fraction_season_2", float),
	"kct_flux_fraction_season_3": ("kct_flux_fraction_season_3", float),
	"kct_snr_season_0": ("kct_snr_season_0", float),
	"kct_snr_season_1": ("kct_snr_season_1", float),
	"kct_snr_season_2": ("kct_snr_season_2", float),
	"kct_snr_season_3": ("kct_snr_season_3", float),
	"kct_distance_0": ("kct_distance_0", int),
	"kct_distance_1": ("kct_distance_1", int),
	"kct_distance_2": ("kct_distance_2", int),
	"kct_distance_3": ("kct_distance_3", int),
	"kct_channel_season_0": ("kct_channel_season_0", int),
	"kct_module_season_0": ("kct_module_season_0", int),
	"kct_output_season_0": ("kct_output_season_0", int),
	"kct_row_season_0": ("kct_row_season_0", int),
	"kct_column_season_0": ("kct_column_season_0", int),
	"kct_channel_season_1": ("kct_channel_season_1", int),
	"kct_module_season_1": ("kct_module_season_1", int),
	"kct_output_season_1": ("kct_output_season_1", int),
	"kct_row_season_1": ("kct_row_season_1", int),
	"kct_column_season_1": ("kct_column_season_1", int),
	"kct_channel_season_2": ("kct_channel_season_2", int),
	"kct_module_season_2": ("kct_module_season_2", int),
	"kct_output_season_2": ("kct_output_season_2", int),
	"kct_row_season_2": ("kct_row_season_2", int),
	"kct_column_season_2": ("kct_column_season_2", int),
	"kct_channel_season_3": ("kct_channel_season_3", int),
	"kct_module_season_3": ("kct_module_season_3", int),
	"kct_output_season_3": ("kct_output_season_3", int),
	"kct_row_season_3": ("kct_row_season_3", int),
	"kct_column_season_3": ("kct_column_season_3", int),
	"ktswcKey": ("ktswcKey", int),
	"NUVg_KIS": ("NUVg_KIS", float),
	"Ug_KIS": ("Ug_KIS", float),
	"gr_KIS": ("gr_KIS", float),
	"gi_KIS": ("gi_KIS", float),
	"rHa_KIS": ("rHa_KIS", float),
	"BV_UBV": ("BV_UBV", float),
	"UB_UBV": ("UB_UBV", float),
	"gJ_KIS_IRT": ("gJ_KIS_IRT", float),
	"ID_UBV": ("ID_UBV", int),
	"ra_UBV": ("ra_UBV", float),
	"dec_UBV": ("dec_UBV", float),
	"U_UBV": ("U_UBV", float),
	"B_UBV": ("B_UBV", float),
	"V_UBV": ("V_UBV", float),
	"Sep_UBV": ("Sep_UBV", float),
	"Sep_UBV_IRT": ("Sep_UBV_IRT", float),
	"Sep_UBV_KIS": ("Sep_UBV_KIS", float),
	"det_UBV": ("det_UBV", int),
	"ra_KIS": ("ra_KIS", float),
	"dec_KIS": ("dec_KIS", float),
	"U_KIS": ("U_KIS", float),
	"g_KIS": ("g_KIS", float),
	"r_KIS": ("r_KIS", float),
	"i_KIS": ("i_KIS", float),
	"Ha_KIS": ("Ha_KIS", float),
	"Sep_KIS": ("Sep_KIS", float),
	"Sep_KIS_IRT": ("Sep_KIS_IRT", float),
	"KIS_ID_KIS": ("KIS_ID_KIS", str),
	"ID_KIS": ("ID_KIS", int),
	"class_U_KIS": ("class_U_KIS", int),
	"class_g_KIS": ("class_g_KIS", int),
	"class_r_KIS": ("class_r_KIS", int),
	"class_i_KIS": ("class_i_KIS", int),
	"class_Ha_KIS": ("class_Ha_KIS", int),
	"condition_flag": ("condition_flag", str),
	"ID_SDSS": ("ID_SDSS", int),
	"RA_SDSS": ("RA_SDSS", float),
	"DEC_SDSS": ("DEC_SDSS", float),
	"u_SDSS": ("u_SDSS", float),
	"g_SDSS": ("g_SDSS", float),
	"r_SDSS": ("r_SDSS", float),
	"i_SDSS": ("i_SDSS", float),
	"z_SDSS": ("z_SDSS", float),
	"err_u_SDSS": ("err_u_SDSS", float),
	"err_g_SDSS": ("err_g_SDSS", float),
	"err_r_SDSS": ("err_r_SDSS", float),
	"err_i_SDSS": ("err_i_SDSS", float),
	"err_z_SDSS": ("err_z_SDSS", float),
	"extinction_u_SDSS": ("extinction_u_SDSS", float),
	"extinction_g_SDSS": ("extinction_g_SDSS", float),
	"extinction_r_SDSS": ("extinction_r_SDSS", float),
	"extinction_i_SDSS": ("extinction_i_SDSS", float),
	"extinction_z_SDSS": ("extinction_z_SDSS", float),
	"ug_SDSS": ("ug_SDSS", float),
	"gr_SDSS": ("gr_SDSS", float),
	"gi_SDSS": ("gi_SDSS", float),
}

periodic_dic = {
    "KID": ("KID", int),
    "Teff": ("Teff", int),
    "logg": ("logg", float),
    "Mass": ("Mass", float),
    "Prot": ("Prot", float),
    "Prot_err": ("Prot_err", float),
    "Rper": ("Rper", float),
    "LPH": ("LPH", float),
    "w": ("w", float),
    "DC": ("DC", int),
    "Flag": ("Flag", str)
}

nonperiodic_dic = {
    "KID": ("KID", int),
    "Teff": ("Teff", int),
    "logg": ("logg", float),
    "Mass": ("Mass", float),
    "Prot": ("Prot", float),
    "Prot_err": ("Prot_err", float),
    "LPH": ("LPH", float),
    "w": ("w", float),
    "DC": ("DC", int),
}

mast_table_dic = {
}

kplr_dic = mast_kic10_dic

