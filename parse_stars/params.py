from settings import setup_logging
from utils import get_nth_row
logger = setup_logging()



# Mathur Q1-17 Table
updated_fields = ["KIC", "Teff", "E_Teff", "e_Teff", "log(g)", "E_log(g)", "e_log(g)", "[Fe/H]", "E_[Fe/H]", "e_[Fe/H]", "Rad", "E_Rad", "e_rad", "Mass", "E_Mass", "e_mass", "rho", "E_rho", "e_rho", "Dist", "E_Dist", "e_Dist", "Av", "E_Av", "e_Av", "Mod"]
updated_types = [int, int, int, int, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, str]

        self.periodic_dir = "./data/Table_Periodic.txt"
        self.unperiodic_dir = "./data/Table_Non_Periodic.txt"
        self.mast_table_dir = "./data/kepler_fov_search_all.txt"


updated_dir = "./data/table4.dat"
updated_dic = {
    "KIC": ("KIC", int),
    "Teff": ("Teff", int),
    "E_Teff": ("E_Teff", int),
    "e_Teff": ("e_Teff", int),
    "log(g)": ,
    "E_log(g)": ,
    "e_log(g)": ,
    "[Fe/H]":,
    "E_[Fe/H]":,
    "e_[Fe/H]":,
    "Rad":,
    "E_Rad":,
    "e_rad":,
    "Mass":,
    "E_Mass":,
    "e_mass":,
    "rho":,
    "E_rho":,
    "e_rho":,
    "Dist":,
    "E_Dist":,
    "e_Dist":,
    "Av":,
    "E_Av":,
    "e_Av":,
    "Mod":
}


mast_dic = {}
periodic_dic = {}
mast_table_dic = {}
kplr_dic = {}

