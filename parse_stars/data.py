# TODO: TO FIX UP

from settings import setup_logging, filename_stellar_params
logger = setup_logging()

"""
Kepler_ID,Master ID,kepmag,Ang Sep ('),RA (J2000),
Dec (J2000),Data Availability,Seasons on CCD,Teff,Log_G,
E(B-V),g,r,J,GALEX NUV,
ID_IRT,J_IRT,gr,gi,gK,
gJ,GALEX FUVNUV,NUVg,Contamination season 0,Flux Fraction season 0,
Edge_Distance_0 (px),Module_0,Module_1,Module_2,Module_3,
Ug_KIS,gr_KIS,gi_KIS,BV_UBV,UB_UBV,
gJ_KIS_IRT,V_UBV,g_KIS,r_KIS,class_g_KIS,
Condition flag,ID_SDSS,u_SDSS,g_SDSS,r_SDSS,
i_SDSS,z_SDSS,ug_SDSS,gr_SDSS,gi_SDSS

"""

# TODO: function
def get_table_params(kics, params, fout, table_file=filename_stellar_params):
    if "Kepler_ID" not in params:
        params = ["Kepler_ID"] + params
    params_list = []
    curr_dicts = []
    with open(table_file) as input_f:
        curr_line = input_f.readline().strip()
        if curr_line[:13] == "Input line 1:": #id line
            fieldnames = input_f.readline().strip().split(',')
            input_f.readline() #types, useless line
            for f_i, fieldname in enumerate(fieldnames):
                if len(params_list) == len(params):
                    break
                for param in params:
                    if fieldname == param:
                        params_list.append((param, f_i))
            if len(params_list) != len(params):
                print(params)
                logger.error("Error: not all params found")
                return 1
        else:
            logger.error("Error: file doesn't have right format")
        for line in input_f:
            if line[:10] == "Input line":
                continue
            curr_dict = {}
            curr_data = line.strip().split(',')
            for par, par_i in params_list:
                curr_dict[par] = curr_data[par_i]
            curr_dicts.append(curr_dict)
    dict_to_file(fout, curr_dicts)
    if len(kics) != len(curr_dicts):
        logger.error("Error: not all kics are processed")
    logger.debug("get_table_params done")
    return 0

# TODO: function
def get_mast_params(target, params):
    data = target.mast_request("kepler", "Kepler_ID")
    print(data)
    return

# TODO: function
def is_faint_table(target, min_kepmag=15, table_file=filename_stellar_params):
    with open(table_file) as input_f:
        curr_line = input_f.readline().strip()
        if curr_line[:13] == "Input line 1:": #id line
            fieldnames = input_f.readline().strip().split(',')
            input_f.readline() #types, useless line
            for f_i, fieldname in enumerate(fieldnames):
                if fieldname == "Kepler_ID":
                    kid_col_no = f_i
                if fieldname == "kepmag":
                    kepmag_col_no = f_i
        if curr_line[kid_col_no] == target:
            return curr_line[kepmag_col_no] > min_kepmag
        for line in input_f:
            if line[:10] == "Input line":
                continue
            curr_data = line.strip().split(',')
            if curr_data[kid_col_no] == target:
                return curr_data[kepmag_col_no] > min_kepmag
    logger.debug("get_table_params done")
    return False
