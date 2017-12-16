import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import logging
import csv
import os
from matplotlib.backends.backend_pdf import PdfPages
from f3 import photometry

# Setup
f3_location = "/home/user/Desktop/astrolab/solar_data/f3_edit/"

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

# fmt = ['ko', 'rD', 'b^', 'gs']

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
        plt.errorbar(targ.times[g], targ.obs_flux[g], yerr=targ.flux_uncert[i], fmt=targ.fmt[i])

    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Relative Flux', fontsize=15)

    plt.subplot2grid((3,3), (1,2))
    plt.title(targ.kic, fontsize=20)
    jj, ii = targ.center
    jj, ii = int(jj), int(ii)
    img = np.sum(((targ.targets == 1)*targ.postcard + (targ.targets == 1)*100000)
                     [:,jj-image_region:jj+image_region,ii-image_region:ii+image_region], axis=0)
    plt.imshow(img, interpolation='nearest', cmap='gray', vmin=98000*52, vmax=104000*52)

    fig.tight_layout()

def run_photometry(targ):
    target = photometry.star(targ)

    target.make_postcard()

    target.find_other_sources(ntargets=100, plot_flag=False)

    plot_data(target)

def plot_targets(targets_file):    
    with PdfPages(output_file) as pdf:
        for targets_file in targets_files:
            with open(targets_file) as f:
                reader = csv.reader(f)
                for count, row in enumerate(reader):
                    target = row[0]
                    run_photometry(target)
                    pdf.savefig()
                    plt.close()
                    logger.info(str(count + 1) + "\t" + target + "\tplot_done")
    logger.info("plot_all_targets_done")


def main():
    plot_targets(targets_file)

if __name__ == "__main__" and __package__ is None:
    from os import sys, path
    sys.path.append(f3_location)
    main()
