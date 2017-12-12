from f3 import photometry
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import logging
import csv

# Setup
file_name = "parsed_single"

targets_file = file_name + ".txt"
output_file = "plot_" + file_name + ".pdf"
log_file = file_name + ".log"

fmt = ['ko', 'rD', 'b^', 'gs']


# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_file)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.getLogger(photometry.__name__).setLevel(logging.ERROR)

# Functions

def plot_data(targ, do_roll=True, ignore_bright=0):
    targ.data_for_target(do_roll, ignore_bright)

    for i in range(4):
        g = np.where(targ.qs == i)[0]
        plt.errorbar(targ.times[g], targ.obs_flux[g], yerr=targ.flux_uncert[i], fmt=fmt[i])

    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Relative Flux', fontsize=15)
    plt.title(targ.kic, fontsize=15)


def run_photometry(targ):
    target = photometry.star(targ)

    target.make_postcard()

    target.find_other_sources()

    plot_data(target)


def plot_targets(targets_file):
    with PdfPages(output_file) as pdf:
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

if __name__ == "__main__":
    main()