from utils import does_path_exists
from settings import setup_logging

logger = setup_logging()

# prints one dict of kid, kepmag, angsep to csv file
def dict_to_file(fout, dicts, keys=None, bypass_prompt=True):
    if not bypass_prompt and does_path_exists(fout):
        return 1
    if keys == None:
        keys = [key for key in dicts[0]][1:]
    with open(fout, 'w') as write_f:
        for key_i in range(len(keys) - 1):
            write_f.write([keys[key_i]] + ",")
            write_f.write([keys[-1]] + "\n")
        for i in range(len(dicts)):
            write_f.write(dicts[i]["Kepler_ID"] + ",")
            for key_i in range(len(keys) - 1):
                write_f.write(dicts[i][keys[key_i]] + ",")
            write_f.write(dicts[i][keys[-1]] + "\n")
    return 0

# prints a dict of a kic and its neighbours to csv file
def dicts_to_file(filename, dicts, keys=None, bypass_prompt=True):
    if not bypass_prompt and does_path_exists(filename):
        return 1
    if keys == None:
        keys = [key for key in dicts[0]][1:]
    with open(filename, 'w') as write_f:
        for i in range(len(dicts)):
            write_f.write("input: " + dicts[i]["Kepler_ID"] + "\n")
            for kids in arr[i]:
                write_f.write(dicts[i]["Kepler_ID"] + ",")
            for key_i in range(len(keys) - 1):
                write_f.write(dicts[i][keys[key_i]] + ",")
            write_f.write(dicts[i][keys[-1]] + "\n")
    return 0
# TODO: TO FIX UP

# helper for remove_bright_neighbours_together()
def element_is_not_in_list(arr, n):
    if len(arr) == 0:
        return True
    for i in range(len(arr)):
        if n == arr[i]:
            return False
    return True

# removes stars from list with bright neighbours, and removes duplicates
# assumes that list of neighbour stars follows each target stars, and
#   list of input stars has same order and is same as processed stars
def remove_bright_neighbours_together(folder, filename_out, difference_max=2.0, bypass_prompt=True):
    all_kids = []
    kepmag_col_no = 1
    curr_id = -1
    count = 0
    # filename = parsed_kids_filename

    if not bypass_prompt and does_path_exists(filename):
        return 1

    input_files = sorted([filename for filename in os.listdir(folder) \
                          if filename.startswith(kepmag_file_prefix)])

    with open(filename_out, 'w') as output_f:
        for input_files_i in range(len(input_files)):
            with open(input_files[input_files_i]) as input_f:
                for line in input_f:
                    curr_line = line.strip()
                    if curr_line[0:10] == "Input line": #id line
                        if curr_line[10:13] == " 1:": # under input 1 is labels
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
                        elif abs(curr_kepmag - float(test_kepmag)) <= difference_max:
                            if element_is_not_in_list(all_kids, test_kid):
                                all_kids.append(test_kid)
                                output_f.write(str(test_kid) + "\n")
                                count += 1
    logger.info("printed " + str(count) + " kids")
    logger.info("remove_bright_neighbours_together done")
    return 0

# doesn't remove duplicates
# outputs separate files for stars with no neighbours and with neighbours
def remove_bright_neighbours_separate(folder, fout_prefix, difference_max=2.0):
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
    fout_single = single_parsed_kids_filename
    fout_batch = batch_parsed_kids_filename

    input_files = sorted([filename for filename in os.listdir(folder) \
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
                            # don't iterate through fieldnames unnecessarily
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
                        curr_dict = {"kepmag": curr_kepmag, "Ang Sep (')": curr_angsep}
                        continue
                    if is_first_entry: # previous star had no neighbours
                        curr_dict["Kepler_ID"] = curr_kid
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
                    index += 1
                    curr_kepmag = float(curr_data[kepmag_col_no])
                    curr_angsep = curr_data[angsep_col_no]
                    curr_dict = {"kepmag": str(curr_kepmag), "Ang Sep (')": curr_angsep}
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
                        test_dict = {"kepmag": str(test_kepmag), "Ang Sep (')": test_angsep}
                        if is_first_entry: # need to intialise dictionary
                            batch_dict = {"Kepler_ID": curr_kid, curr_kid: curr_dict, \
                                          test_kid: test_dict}
                            is_first_entry = False
                        else:
                            batch_dict[test_kid] = test_dict
            if not_passed_last_entry:
                if is_first_entry:
                    curr_dict["Kepler_ID"] = curr_kid
                    single_kids.append(curr_dict)
                else:
                    batch_kids.append(batch_dict)
                not_passed_last_entry = False
    dict_to_file(fout_single, single_kids)
    dicts_to_file(fout_batch, batch_kids)
    logger.info("printed " + str(len(single_kids)) + " kids with no neighbours")
    logger.info("printed " + str(len(batch_kids)) + " kids with neighbours")
    logger.info("remove_bright_neighbours_separate done")
    return 0
