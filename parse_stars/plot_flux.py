import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import logging
import csv
import os
from matplotlib.backends.backend_pdf import PdfPages

# Setup
f3_location = "/home/user/Desktop/astrolab/solar_data/parse_stars/f3"

file_name = "sample_single"
targets_files = [file_name]

targets_files_temp = ["./data/" + x.split('.')[0]
                      for x in os.listdir("./data/")
                      if x.endswith(".txt") and "single" in x]
if len(targets_files) != 0:
    file_name = targets_files_temp[0]
    targets_files = targets_files_temp

targets_file = file_name + ".txt"
output_file = file_name + "_plot.pdf"
log_file = file_name + ".log"

# Logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Functions

def plot_data(targ, do_roll=True, ignore_bright=0, image_region=15):

    targ.data_for_target(do_roll, ignore_bright)

    fig = plt.figure(figsize=(11,8))
    gs.GridSpec(3,3)

    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    for i in range(4):
        g = np.where(targ.qs == i)[0]
        print(i)
        print(g)
        plt.errorbar(targ.times[g], targ.obs_flux[g], yerr=targ.flux_uncert[i], fmt=targ.fmt[i])

    print(end)
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Relative Flux', fontsize=15)

    plt.subplot2grid((3,3), (1,2))
    plt.title(targ.kic, fontsize=20)
    jj, ii = targ.center
    jj, ii = int(jj), int(ii)
    img = np.sum(((targ.targets == 1)*targ.postcard + (targ.targets == 1)*100000)
                     [:,jj-image_region:jj+image_region,ii-image_region:ii+image_region], axis=0)
    plt.imshow(img, interpolation='nearest', cmap='gray', vmin=98000*52, vmax=104000*52)

    fig.text(7.5/8.5, 0.5/11., str(count + 1), ha='center', fontsize = 12)
    fig.tight_layout()

def run_photometry(targ):
    target = photometry.star(targ)

    target.make_postcard()

    target.find_other_sources(ntargets=100, plot_flag=False)

    plot_data(target)

def plot_targets(files):
    with PdfPages(output_file) as pdf:
        for single_file in files:
            targets_file = single_file + ".txt"
            with open(targets_file) as f:
                reader = csv.reader(f)
                for count, row in enumerate(reader):
                    target = row[0]
                    run_photometry(target)
                    pdf.savefig()
                    plt.close()
                    logger.info(str(count + 1) + "\t" + target + "\tplot_done")
    logger.info("plot_all_targets_done")

def is_more_or_less(target, quarter):
    for indexes in quarter:
        is_incr = False
        is_decr = False
        for i in range(len(indexes - 2)):
            if target.obs_flux[indexes[i + 1]] > target.obs_flux[indexes[i]]:
                if is_decr:
                    break
                is_incr = True
            elif target.obs_flux[indexes[i + 1]] < target.obs_flux[indexes[i]]:
                if is_incr:
                    break
                is_decr = True
        return 1 if is_incr else -1 if is_decr else 0

def is_large_ap(target):
    golden = range(0, 8)
    blacks = [[8, 9], [20, 21, 22], [31, 32, 33], [43, 44, 45]]
    reds = [[10, 11, 12], [23, 24, 25], [34, 35, 36], [46, 47, 48]]
    blues = [[13, 14, 15, 16], [26, 27], [37, 38, 39], [49, 50, 51]]
    greens = [[17, 18, 19], [28, 29, 30], [40, 41, 42]]
    channels = [blacks, reds, blues, greens]
    for channel in channels:
        curr_ch = []
        is_strange = False
        for qtr_i in range(len(channel) - 2):
            if channel[qtr_i + 1] is not channel[qtr]:
                break
            else:
                is_strange = True
    return is_strange


def categorise(target):
    categories = ["large_ap", "small_ap", "normal"]
    return is_large_ap(target)

def main():
    #plot_targets(targets_files)
    run_photometry(892376)

if __name__ == "__main__" and __package__ is None:
    from os import sys, path
    sys.path.append(f3_location)
    from f3 import photometry
    main()
