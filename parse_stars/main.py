from settings import *
from utils import *
from aperture import run_photometry, calculate_better_aperture
from data import new_stars
import os
import numpy as np
import csv
from plot import plot_data, print_lc_improved
import matplotlib.pyplot as plt
import itertools

def make_sound(duration=0.3, freq=440):
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    return 0

def fix_things(failed_kics, fix_kics):
    print_lc_improved(failed_kics, ("./results/failed_img.out", \
                                    "./results/failed_old.out", "./results/failed_new.old"))
    logger.info("failed kics done")
    print_lc_improved(fix_kics, ("./results/fix_img.out", \
                                 "./results/fix_old.out", "./results/fix_new.out"))
    logger.info("fixed kics done")

def from_server():
    t = table_api("/data/ffidata/results/lc_data_img.out", ",", 1, None)
    all_t = t.parse_table_arrs([0] + range(5, 905), None, [int] + [float]*900)
    all_t_np = np.array(all_t)
    print all_t_np.shape
    print all_t_np[0]
    count = 0
    succ = 0
    for row in all_t_np:
        if not np.any(row[1:]):
            count += 1
            print row[0]
        else:
            succ += 1
    print count, succ

def replace_lines_fix(fin_old, fin_new, fout):
    fix_kics = get_kics(fin_new, ",", skip_rows=1)
    count_fix = 0
    count_nonfix = 0
    with open(fin_new, "r") as fin:
        reader = csv.reader(fin, delimiter=",", skipinitialspace=True)
        all_data_new = list(reader)
    with open(fin_old, "r") as fio, open(fout, "wb") as fo:
        w = csv.writer(fo, delimiter=",")
        r = csv.reader(fio, delimiter=",", skipinitialspace=True)
        for icount, row in enumerate(r):
            if row[0] in fix_kics:
                i = fix_kics.index(row[0])
                w.writerow(all_data_new[i+1])
                count_fix += 1
            else:
                w.writerow(row)
                count_nonfix += 1
            print icount, "done"
    print count_fix, count_nonfix, count_fix+count_nonfix
    return 0

def replace_all_channels(fin_new, fin_old, fout):
    fix_kics = get_kics(fin_new, ",", 0)
    old_kics = get_kics(fin_old, ",", 1)
    if fix_kics != old_kics:
        print "ERROR DIFFERENT KICS"
        return 1

    with open(fin_new, "r") as fin, open(fin_old, "r") as fio, open(fout, "wb") as fo:
        w = csv.writer(fo, delimiter=",", lineterminator="\n")
        fo.write(next(fio).strip() + ",Note\n")
        rn = csv.reader(fin, delimiter=",", skipinitialspace=True)
        ro = csv.reader(fio, delimiter=",", skipinitialspace=True)
        index = 0
        for rown, rowo in itertools.izip(rn, ro):
            #check
            if rown[0] != rowo[0]:
                print "ERROR KICS NOT IN ORDER", rown[0], rowo[0]
                return 1

            new_arr = [rown[0]] + rown[1:3] + rowo[3:] + [rown[-1]]
            w.writerow(new_arr)
            index += 1
            print rown[0], index
    print "replaced all", index
    return 0

def print_new_channels(kics, fout):
    successes = 0
    failed = 0
    with open(fout, "r") as f:
        w = csv.writer(f, delimiter=',', lineterminator='\n')
        for icount, kic in enumerate(kics):
            target = run_photometry(kic)
            if target == 1:
                w.writerow([kic,"FAILED","--"])
                failed += 1
                continue
            targ = target.target
            col = [targ.params['Column_0'], targ.params['Column_1'],
                   targ.params['Column_2'], targ.params['Column_3']]
            row = [targ.params['Row_0'], targ.params['Row_1'],
                   targ.params['Row_2'], targ.params['Row_3']]

            for i in range(len(col)):
                if col[i] is None or row[i] is None:
                    col[i] = np.nan
                    row[i] = np.nan

            center = [str(row), str(col)]
            concat = [np.nanmin(row), np.nanmin(col), abs(shape[0]-np.nanmax(row)), \
                      abs(shape[1]-np.nanmax(col))]
            if is_n_bools(concat, 1, lambda x: x <= min_distance):
                flag = "Close to edge"
            else:
                flag = "--"

            w.writerow([kic] + [center] + [flag])
            successes += 1
            print kic, icount
    print "new channels", successes, failed
    return 0

def main():
    logger.info("### starting ###")
    np.set_printoptions(linewidth=1000) #, precision=4)

    # replace_lines_fix("./tests/old.txt", "./tests/new.txt", "./tests/dat.txt")

    # base_params = ["teff", "logg", "metallicity", "rad", "mass" "rho", "dist", "av", \
    #                "periodic"]
    # param_dic = {"neighbours": [lambda x: x != []]}
    # kics = ["757280", "757450"]
    # n = new_stars(kics)
    # n.get_basic_params(0.15)
    # print n.res

    # target = run_photometry("8754750", plot_flag=False)
    # calculate_better_aperture(target, 0.001, 2, 0.7, 15)
    # plot_data(target)
    # plt.show()

    # replace_all_channels("./tests/new.txt", "./tests/old.txt", "./tests/bak.txt")

    make_sound(0.8, 440)
    logger.info("### everything done ###")
    return 0

if __name__ == "__main__":
    os.sys.path.append(f3_location)
    from f3 import photometry

    from settings import setup_main_logging
    logger = setup_main_logging()

    mpl_setup()
    main()


