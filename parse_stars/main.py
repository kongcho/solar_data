"""
TODO:
- move photometry stuff somewhere
- please clean up :'( don't import *
"""

import os
import csv
import itertools
import numpy as np

import matplotlib
if os.environ.get('DISPLAY','') == "":
    print("Using non-interactive Agg backend")
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from matplotlib.backends.backend_pdf import PdfPages

import lightkurve as lk
from astropy.modeling import models, fitting

from utils import *
from settings import *
from aperture import *
from data import *
from parse import *
from plot import *

logger = setup_main_logging()

## Functions
def model_background(target, coords, model_pix=15, max_factor=0.2):
    for i in range(target.postcard.shape[0]):
        min_i, max_i, min_j, max_j = coords
        region = target.postcard[i]
        img = region[min_i:max_i, min_j:max_j]

        mask = make_background_mask_max(target, img, model_pix, max_factor)
        z = np.ma.masked_array(img, mask=mask)
        img -= np.ma.median(z)

    target.integrated_postcard = np.sum(target.postcard, axis=0)
    target.data_for_target(do_roll=True, ignore_bright=0)
    return target

def logical_or_all_args(*args):
    result = np.zeros_like(args[0])
    for arg in args:
        result += arg
    return np.where(result != 0, 1, 0)

def make_background_mask_max(target, img, model_pix=15, max_factor=0.2):
    if not np.any(img):
        return -1

    coords = clip_array([target.center[0]-model_pix, target.center[0]+model_pix, \
                         target.center[1]-model_pix, target.center[1]+model_pix], \
                        [0, target.postcard.shape[1]-1, 0, target.postcard.shape[2]-1], \
                        [False, True, False, True])
    min_i, max_i, min_j, max_j = coords

    max_mask = np.where(img >= np.percentile(img, int((1-max_factor)*100)), 1, 0)
    targets_mask = np.where(target.targets != 0, 1, 0)[min_i:max_i, min_j:max_j]
    mask = logical_or_all_args(max_mask, targets_mask)

    return mask

def plot_box(x1, x2, y1, y2, marker='r-', **kwargs):
    plt.plot([x1, x1], [y1, y2], marker, **kwargs)
    plt.plot([x2, x2], [y1, y2], marker, **kwargs)
    plt.plot([x1, x2], [y1, y1], marker, **kwargs)
    plt.plot([x1, x2], [y2, y2], marker, **kwargs)

def testing(targ, fout="./", image_region=15, model_pix=15, \
            mask_factor=0.001, max_factor=0.2, save_pdf=True):

    target = photometry.star(targ, ffi_dir=ffidata_folder)

    try:
        target.make_postcard()
    except Exception as e:
        logger.info("run_photometry unsuccessful: %s" % target.kic)
        logger.error(e.message)
        return 1

    run_partial_photometry(target)

    coords = clip_array([target.center[0]-model_pix, target.center[0]+model_pix, \
                         target.center[1]-model_pix, target.center[1]+model_pix], \
                        [0, target.postcard.shape[1]-1, 0, target.postcard.shape[2]-1], \
                        [False, True, False, True])
    min_i, max_i, min_j, max_j = coords

    # make mask to improve aperture
    tar = target.target
    channel = [tar.params['Channel_0'], tar.params['Channel_1'],
               tar.params['Channel_2'], tar.params['Channel_3']]

    kepprf = lk.KeplerPRF(channel=channel[0], shape=(image_region*2, image_region*2), \
                          column=image_region, row=image_region)
    prf = kepprf(flux=1000, center_col=image_region*2, center_row=image_region*2, \
                 scale_row=1, scale_col=1, rotation_angle=0)
    mask = np.where(prf > mask_factor*np.max(prf), 1, 0)

    with PdfPages(fout + targ + "_out.pdf") as pdf:
        fig0 = plot_data(target)
        plt.gcf().text(4/8.5, 1/11., str(np.nanmean(target.flux_uncert)), \
                       ha='center', fontsize = 11)
        pdf.savefig()
        plt.close(fig0)

        improve_aperture(target, mask, image_region, relax_pixels=2)
        fig0 = plot_data(target)
        plt.gcf().text(4/8.5, 1/11., str(np.nanmean(target.flux_uncert)), \
                       ha='center', fontsize = 11)
        pdf.savefig()
        plt.close(fig0)

        model_background(target, coords, model_pix, max_factor)

        fig0 = plot_data(target)
        plt.gcf().text(4/8.5, 1/11., str(np.nanmean(target.flux_uncert)), \
                       ha='center', fontsize = 11)
        pdf.savefig()
        plt.close("all")

        return target

def plot_background_modelling(targ, fout="./", image_region=15, model_pix=15, mask_factor=0.001, \
                              max_factor=0.2, min_img=-1000, max_img=1000, save_pdf=True):

    target = photometry.star(targ, ffi_dir=ffidata_folder)

    try:
        target.make_postcard()
    except Exception as e:
        logger.info("run_photometry unsuccessful: %s" % target.kic)
        logger.error(e.message)
        return 1

    run_partial_photometry(target)

    # make temp variables
    save_post = np.empty_like(target.postcard)
    save_post[:] = target.postcard

    save_int = np.empty_like(target.integrated_postcard)
    save_int[:] = target.integrated_postcard

    coords = clip_array([target.center[0]-model_pix, target.center[0]+model_pix, \
                         target.center[1]-model_pix, target.center[1]+model_pix], \
                        [0, target.postcard.shape[1]-1, 0, target.postcard.shape[2]-1], \
                        [False, True, False, True])
    min_i, max_i, min_j, max_j = coords

    # plot
    fig1 = plt.figure(1, figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(save_int, interpolation='nearest', cmap='gray', \
               vmin=min_img, vmax=max_img, origin='lower')
    plot_box(min_j, max_j, min_i, max_i, 'r-', linewidth=1)

    # make mask to improve aperture
    tar = target.target
    channel = [tar.params['Channel_0'], tar.params['Channel_1'],
               tar.params['Channel_2'], tar.params['Channel_3']]

    kepprf = lk.KeplerPRF(channel=channel[0], shape=(image_region*2, image_region*2), \
                          column=image_region, row=image_region)
    prf = kepprf(flux=1000, center_col=image_region*2, center_row=image_region*2, \
                 scale_row=1, scale_col=1, rotation_angle=0)
    mask = np.where(prf > mask_factor*np.max(prf), 1, 0)

    with PdfPages(fout + targ + "_out.pdf") as pdf:
        # original plot
        fig0 = plot_data(target)
        plt.gcf().text(4/8.5, 1/11., str(np.nanmean(target.flux_uncert)), \
                       ha='center', fontsize = 11)
        pdf.savefig()
        plt.close(fig0)

        # improved aperture plot
        improve_aperture(target, mask, image_region, relax_pixels=2)
        fig0 = plot_data(target)
        plt.gcf().text(4/8.5, 1/11., str(np.nanmean(target.flux_uncert)), \
                       ha='center', fontsize = 11)
        pdf.savefig()
        plt.close(fig0)

        for i in range(target.postcard.shape[0]):
            # make model
            region = target.postcard[i]
            img = region[min_i:max_i, min_j:max_j]

            mask = make_background_mask_max(target, img, model_pix, max_factor)
            z = np.ma.masked_array(img, mask=mask)
            img -= np.ma.median(z)

            if i == 0:
                n = 3
                fig2 = plt.figure(2, figsize=(10, 4))

                plt.subplot(1, n, 1)
                plt.imshow(mask, cmap='gray', vmin=0, vmax=1, origin='lower')
                plt.title("Mask")

                plt.subplot(1, n, 2)
                plt.imshow(z, interpolation='nearest', cmap='gray', \
                           vmin=-200, vmax=1000, origin='lower')
                plt.title("Data")

                plt.subplot(1, n, 3)
                plt.imshow(img, interpolation='nearest', cmap='gray', \
                           vmin=-200, vmax=1000, origin='lower')
                plt.title("Residual")
                plt.colorbar()

                if save_pdf:
                    pdf.savefig()
                    plt.close(fig2)

        # finalise new postcard
        target.integrated_postcard = np.sum(target.postcard, axis=0)
        target.data_for_target(do_roll=True, ignore_bright=0)

        # plot rest of stuff
        plt.figure(1)
        plt.subplot(1, 2, 2)
        plt.imshow(target.integrated_postcard, interpolation='nearest', cmap='gray', \
                   vmin=min_img, vmax=max_img, origin='lower')
        plot_box(min_j, max_j, min_i, max_i, 'r-', linewidth=1)
        plt.colorbar()

        if save_pdf:
            pdf.savefig()
            plt.close(fig1)
        else:
            plt.show()
        plt.close("all")

        fig0 = plot_data(target)
        plt.gcf().text(4/8.5, 1/11., str(np.nanmean(target.flux_uncert)), \
                       ha='center', fontsize = 11)
        pdf.savefig()
        plt.close("all")

        return target

def format_arr(arr, sep="\t"):
    return sep.join(str(i) for i in arr)

def make_sound(duration=0.3, freq=440):
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    return 0

def main():
    logger.info("### starting ###")
    ## gets good kics to be put into MAST
    # array_to_file(get_good_existing_kids(filename_stellar_params, get_kics(filename_periods)))

    ## uses MAST results to get stars without neighbours
    # remove_bright_neighbours_separate()

    ## plots list of files with kics using f3
    # plot_targets(targets_file, [is_large_ap, has_peaks], get_kics(targets_file, ' ', 0))


    ## TEMP
    ben_kics = ["2694810"
                , "3236788"
                , "3743810"
                , "4555566"
                , "4726114"
                , "5352687"
                , "5450764"
                , "6038355"
                , "6263983"
                , "6708110"
                , "7272437"
                , "7432092"
                , "7433192"
                , "7678238"
                , "8041424"
                , "8043142"
                , "8345997"
                , "8759594"
                , "8804069"
                , "9306271"
                , "10087863"
                , "10122937"
                , "11014223"
                , "11033434"
                , "11415049"
                , "11873617"
                , "12417799"
                ]

    ## TESTS
    np.set_printoptions(linewidth=1000, precision=1)

    # kics = (get_nth_kics(filename_stellar_params, 4000, 1, ' ', 0))[:]
    kics = ["11913365", "11913377"] + ben_kics

    # SIMPLE TESTS
    for kic in kics:
        plot_background_modelling(kic, save_pdf=True, fout="./", model_pix=15, max_factor=0.2)
        # testing(kic, save_pdf=True, fout="./", model_pix=15, max_factor=0.2)

    make_sound(0.3, 440)
    logger.info("### everything done ###")
    return 0

if __name__ == "__main__":
    os.sys.path.append(f3_location)
    from f3 import photometry
    main()
