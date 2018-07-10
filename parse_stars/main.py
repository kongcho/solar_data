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

# from utils import clip_array
# from settings import setup_logging, f3_location, ffidata_folder
# from plot import plot_data

from settings import *
from aperture import *
from data import *
from parse import *
from plot import *

logger = setup_logging()

## Functions

# helper function for different functions
# runs find_other_sources under different parameters to change the aperture
def run_partial_photometry(target, image_region=15, edge_lim=0.015, min_val=5000, ntargets=100, \
                           extend_region_size=3, remove_excess=4, plot_window=15, plot_flag=False):

    try:
        target.find_other_sources(edge_lim, min_val, ntargets, extend_region_size, \
                              remove_excess, plot_flag, plot_window)
    except Exception as e:
        logger.info("run_partial_photometry unsuccessful: %s" % target.kic)
        logger.error(e, exc_info=True)
        return 1

    target.data_for_target(do_roll=True, ignore_bright=0)

    jj, ii = target.center
    jj, ii = int(jj), int(ii)

    img = np.sum(((target.targets == 1)*target.postcard + (target.targets == 1)*100000)\
                 [:,jj-image_region:jj+image_region, ii-image_region:ii+image_region], axis=0)

    if (img.shape != (image_region*2, image_region*2)):
        sides = []
        if jj + image_region > target.targets.shape[0]:
            sides += "Top"
        if ii + image_region > target.targets.shape[1]:
            sides += "Left"
        img = pad_img_wrap(img, (image_region*2, image_region*2), sides)

    setattr(photometry.star, 'img', img)

    logger.info("run_partial_photometry done: %s" % target.kic)
    return target

# sets up photometry for a star and adds aperture to class
def run_photometry(targ, image_region=15, edge_lim=0.015, min_val=5000, ntargets=100, \
                   extend_region_size=3, remove_excess=4, plot_window=15, plot_flag=False):

    try:
        target = photometry.star(targ, ffi_dir=ffidata_folder)
        target.make_postcard()
    except Exception as e:
        logger.info("run_photometry unsuccessful: %s" % target.kic)
        logger.error(e.message)
        return 1

    return run_partial_photometry(target, image_region, edge_lim, min_val, ntargets, \
                                  extend_region_size, remove_excess, plot_window, plot_flag)

# outputs dict of functions that finds faulty stars
#   and kics that fall in those functions
def get_boolean_stars(targets, boolean_funcs, edge_lim=0.015, min_val=500, ntargets=100):
    full_dict = {}
    full_dict["good"] = []
    for boolean_func in boolean_funcs:
        full_dict[boolean_func.__name__] = []
    for targ in targets:
        is_faulty = False
        target = run_photometry(targ, edge_lim=edge_lim, min_val=min_val, ntargets=ntargets)
        if target == 1:
            return 1
        for boolean in boolean_funcs:
            if boolean(target):
                full_dict[boolean.__name__].append(target)
                is_faulty = True
        if not is_faulty:
            full_dict["good"].append(target)
    logger.info("get_boolean_stars done")
    return full_dict


def make_model_background(img, model_pix=15):
    ycoord = min(model_pix*2, img.shape[0])
    xcoord = min(model_pix*2, img.shape[1])
    y, x = np.mgrid[:ycoord, :xcoord]
    p_init = models.Polynomial2D(degree=2)
    fit_p = fitting.LinearLSQFitter()
    p = fit_p(p_init, x, y, z=img)

    model = p(x, y)
    # model = np.clip(model, np.min(z), np.max(z))

    return model

def model_background(target, model_pix):
    for i in range(target.postcard.shape[0]):
        region = target.postcard[i]
        coords = clip_array([target.center[0]-model_pix, target.center[0]+model_pix, \
                             target.center[1]-model_pix, target.center[1]+model_pix], \
                            [0, target.postcard.shape[1]-1, 0, target.postcard.shape[2]-1], \
                            [False, True, False, True])
        min_i, max_i, min_j, max_j = coords
        img = region[min_i:max_i, min_j:max_j]
        model = make_model_background(img)
        region[min_i:max_i, min_j:max_j] = region[min_i:max_i, min_j:max_j] - model

    target.integrated_postcard = np.sum(target.postcard, axis=0)
    run_partial_photometry(target)
    return target

def get_median_region(arr):
    minimum = np.median(arr)
    return np.where(arr > minimum, 1, 0)

def logical_and_all_args(*args):
    result = []
    for arg in args:
        result =+ arg
    return np.where(result != 0, 1, 0)

def make_background_mask_max(target, img, model_pix=15, max_factor=0.1):
    if not np.any(img):
        return -1

    coords = clip_array([target.center[0]-model_pix, target.center[0]+model_pix, \
                         target.center[1]-model_pix, target.center[1]+model_pix], \
                        [0, target.postcard.shape[1]-1, 0, target.postcard.shape[2]-1], \
                        [False, True, False, True])
    min_i, max_i, min_j, max_j = coords

    maximum = np.max(img[np.nonzero(img)])
    minimum = np.min(img[np.nonzero(img)])

    max_mask = np.where(img >= (maximum-minimum)*max_factor, 1, 0)
    # max_mask = get_median_region(img)
    targets_mask = np.where(target.targets != 0, 1, 0)[min_i:max_i, min_j:max_j]
    mask = logical_and_all_args(max_mask, targets_mask)

    return mask

def build_mask_layer(img, coord, pix):
    i, j = coord
    region = img[i-pix:i+pix, j-pix:j+pix]
    mask_region = get_median_region(region)
    new = np.zeros_like(img)
    new[i-pix:i+pix, j-pix:j+pix] = mask_region
    return new

def make_background_mask_filter(target, img, mask_pixels=2):
    data = img

    # from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage import img_as_float

    im = img_as_float(data)
    # image_max = ndi.maximum_filter(im, size=5, mode='constant')
    coordinates = peak_local_max(im, min_distance=2)
    mask_peaks = np.zeros_like(im)

    for coord in coordinates:
        coord_tuple = tuple(coord)
        mask_peaks += build_mask_layer(im, coord_tuple, mask_pixels)

    mask_peaks = np.where(mask_peaks > 0, 1, 0)
    mask = mask_peaks
    # mask = np.where(image_max >= 0.1*np.average(image_max), 1, 0)

    return mask

def testing(targ):
    image_region = 15
    mask_factor = 0.001
    fout = "./"
    model_pix = 15
    veen = -1000
    veep = 1000
    wanna_save = False

    target = photometry.star(targ, ffi_dir=ffidata_folder)

    try:
        target.make_postcard()
    except Exception as e:
        logger.info("run_photometry unsuccessful: %s" % target.kic)
        logger.error(e.message)
        return 1

    old_post = np.empty_like(target.postcard)
    old_post[:] = target.postcard

    old_int = np.empty_like(target.integrated_postcard)
    old_int[:] = target.integrated_postcard

    post = target.integrated_postcard
    maximum = np.max(post[np.nonzero(post)])
    minimum = 0

    ai, bi, aj, bj = target.center[0]-model_pix, target.center[0]+model_pix, \
                     target.center[1]-model_pix, target.center[1]+model_pix

    coords = clip_array([target.center[0]-model_pix, target.center[0]+model_pix, \
                         target.center[1]-model_pix, target.center[1]+model_pix], \
                        [0, target.postcard.shape[1]-1, 0, target.postcard.shape[2]-1], \
                        [False, True, False, True])
    min_i, max_i, min_j, max_j = coords

    old_int[min_i,min_j:max_j] = maximum
    old_int[max_i,min_j:max_j] = maximum
    old_int[min_i:max_i,min_j] = maximum
    old_int[min_i:max_i,max_j] = maximum

    fig1 = plt.figure(1, figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(old_int, interpolation='nearest', cmap='gray', vmin=veen, vmax=veep, origin='lower')

    run_partial_photometry(target)

    hey_mask = np.where(target.targets != 0, 1, 0)

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

        ranges = []
        mins = []
        maxs = []

        for i in range(target.postcard.shape[0]):
            region = target.postcard[i]

            coords = clip_array([target.center[0]-model_pix, target.center[0]+model_pix, \
                                 target.center[1]-model_pix, target.center[1]+model_pix], \
                                [0, target.postcard.shape[1]-1, 0, target.postcard.shape[2]-1], \
                                [False, True, False, True])
            min_i, max_i, min_j, max_j = coords
            zold = region[min_i:max_i, min_j:max_j]

            wow_mask = make_background_mask_filter(target, zold)

            z = np.ma.masked_array(zold, mask=wow_mask)

            model = make_model_background(z)
            print i, np.ptp(z), np.min(z), np.max(z)
            print i, np.ptp(model), np.min(model), np.max(model)
            ranges.append(np.ptp(model))
            mins.append(np.min(model))
            maxs.append(np.max(model))

            if i == 0:
                fig2 = plt.figure(2, figsize=(8, 2.5))
                plt.subplot(1, 4, 1)
                plt.imshow(wow_mask, cmap='gray', vmin=0, vmax=1, origin='lower')
                plt.title("Mask")

                plt.subplot(1, 4, 2)
                plt.imshow(z,  interpolation='nearest', cmap='gray', vmin=-200, vmax=1000, origin='lower')
                plt.title("Data")
                plt.subplot(1, 4, 3)
                plt.imshow(model,  interpolation='nearest', cmap='gray', vmin=-200, vmax=1000, origin='lower')
                plt.title("Model")
                plt.subplot(1, 4, 4)
                plt.imshow(z - model, interpolation='nearest', cmap='gray', vmin=-200, vmax=1000, origin='lower')
                plt.title("Residual")
                plt.colorbar()
                if wanna_save:
                    pdf.savefig()
                    plt.close(fig2)

            region[min_i:max_i, min_j:max_j] = region[min_i:max_i, min_j:max_j] - model

        print "HEY", targ, np.ptp(ranges), np.ptp(mins), np.ptp(maxs)

        target.integrated_postcard = np.sum(target.postcard, axis=0)
        target.data_for_target(do_roll=True, ignore_bright=0)

        fako = np.zeros_like(target.integrated_postcard)
        fako[:] = target.integrated_postcard
        fako[min_i,min_j:max_j] = maximum
        fako[max_i,min_j:max_j] = maximum
        fako[min_i:max_i,min_j] = maximum
        fako[min_i:max_i,max_j] = maximum

        plt.figure(1)
        plt.subplot(1, 2, 2)
        plt.imshow(fako, interpolation='nearest', cmap='gray', vmin=veen, vmax=veep, origin='lower')
        plt.colorbar()
        if wanna_save:
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
    ben_random = ["8462852"
                  , "3100219"
                  , "7771531"
                  , "9595725"
                  , "9654240"
                  , "6691114"
                  , "7109052"
                  , "8043142"
                  , "8544875"
                  , "9152469"
                  , "9762293"
                  , "11447772"
                  , "9210192"
                  , "1161620"
                  ]

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

    bad_targs = ["893033"
                 , "1161620"
                 , "1162635"
                 , "1164102"
                 , "1293861"
                 , "1295289"
                 , "1430349"
                 , "1431060"
                 , "1162715"
                 , "1295069"
                 , "1433899"
                 ]

    ## TESTS
    np.set_printoptions(linewidth=1000, precision=1)

    kics1 = ["8527137", "8398294", "8397644", "8398286", "8398452", "10122937", "11873617", "3116513", "3116544", "3124279", "8381999"]
    # kics = (get_nth_kics(filename_stellar_params, 4000, 1, ' ', 0))[:]
    # print_lc_improved_aperture(kics, "out.csv")
    kics2 = ["8462852", "8115021", "8250547", "8250550", "8381999", "9091942"]
    kics = ["11913365", "11913377"] + ben_kics

    for kic in kics:
        # target = print_better_aperture(kic)
        testing(kic)

    make_sound(0.3, 440)
    logger.info("### everything done ###")
    return 0

if __name__ == "__main__": # and __package__ is None:
    os.sys.path.append(f3_location)
    from f3 import photometry
    main()
