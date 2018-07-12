"""
TODO:
- move photometry stuff somewhere
- please clean up :'( don't import *
- clean up testing + move to plot.py
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

def make_model_background(img, model_pix=15):
    ycoord = min(model_pix*2, img.shape[0])
    xcoord = min(model_pix*2, img.shape[1])
    y, x = np.mgrid[:ycoord, :xcoord]
    p_init = models.Polynomial2D(degree=2)
    fit_p = fitting.LinearLSQFitter()
    p = fit_p(p_init, x, y, z=img)

    model = p(x, y)
    # model = np.clip(model, np.min(img), np.max(img))

    return model

# TODO: NEED TO UPDATE LMAOOOO
# TODO: WHERE THE MASK AT
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

def logical_or_all_args(*args):
    result = np.zeros_like(args[0])
    for arg in args:
        result += arg
    return np.where(result != 0, 1, 0)

def make_background_mask_max(target, img, model_pix=15, max_factor=0.01):
    if not np.any(img):
        return -1

    coords = clip_array([target.center[0]-model_pix, target.center[0]+model_pix, \
                         target.center[1]-model_pix, target.center[1]+model_pix], \
                        [0, target.postcard.shape[1]-1, 0, target.postcard.shape[2]-1], \
                        [False, True, False, True])
    min_i, max_i, min_j, max_j = coords

    max_mask = np.where(img >= max_factor*np.max(img), 1, 0)
    targets_mask = np.where(target.targets != 0, 1, 0)[min_i:max_i, min_j:max_j]
    mask = logical_or_all_args(max_mask, targets_mask)

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

def arr_within_bound_n_times(arr, n, bound):
    return is_n_bools(arr, n, lambda x: x <= bound)

def check_postcard_ranges(data, postcard, percentile=20):
    """
    want to return true/should change if:
    - the model varies, ie minimum/maximum/avg varies
      - aka the range of mins is higher than boundary
      - boundary as some factor * average mins?
    false/don't change if:
    - model range for each postcard is too low
      - aka the difference in one image is low in comparison to data?
      - data = average masked background? range of background?
    """
    ranges = []
    mins = []
    maxs = []
    for i in range(postcard.shape[0]):
        curr_card = postcard[i]
        ranges.append(np.ptp(curr_card))
        mins.append(np.min(curr_card))
        maxs.append(np.max(curr_card))
    booleans = [is_n_bools(ranges, 52, lambda x: x == 0)
                # , arr_within_range_n_times(mins, 1, np.percentile(mins, percentile))
                # , arr_within_range_n_times(maxs, 1, np.percentile(maxs, percentile))
    ]
    return all(booleans)

def testing(targ, fout="./", image_region=15, model_pix=15, mask_factor=0.001, min_img=-1000, max_img=1000, save_pdf=True):

    target = photometry.star(targ, ffi_dir=ffidata_folder)

    try:
        target.make_postcard()
    except Exception as e:
        logger.info("run_photometry unsuccessful: %s" % target.kic)
        logger.error(e.message)
        return 1

    # make temp vars
    old_post = np.empty_like(target.postcard)
    old_post[:] = target.postcard

    old_int = np.empty_like(target.integrated_postcard)
    old_int[:] = target.integrated_postcard

    post = target.integrated_postcard
    maximum = np.max(post[np.nonzero(post)])

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

    new_post = np.zeros_like(target.postcard)
    new_post[:] = target.postcard[:]

    models = np.zeros((target.postcard.shape[0], max_i-min_i, max_j-min_j))

    # plot
    fig1 = plt.figure(1, figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(old_int, interpolation='nearest', cmap='gray', vmin=min_img, vmax=max_img, origin='lower')

    run_partial_photometry(target)

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

        old_uncerts = np.zeros_like(target.flux_uncert)
        old_uncerts[:] = target.flux_uncert

        data_ranges = []

        for i in range(target.postcard.shape[0]):

            # make model
            region = target.postcard[i]
            z_old = region[min_i:max_i, min_j:max_j]

            mask = make_background_mask_max(target, z_old)
            z = np.ma.masked_array(z_old, mask=mask)
            model = make_model_background(z)
            models[i] = model

            data_ranges.append(np.ptp(z))

            if i == 0:
                fig2 = plt.figure(2, figsize=(8, 2.5))
                plt.subplot(1, 4, 1)
                plt.imshow(mask, cmap='gray', vmin=0, vmax=1, origin='lower')
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
                if save_pdf:
                    pdf.savefig()
                    plt.close(fig2)

            new_post[i, min_i:max_i, min_j:max_j] = region[min_i:max_i, min_j:max_j] - model

        # finalise new postcard
        calculated_int_post = np.zeros_like(target.integrated_postcard)
        calculated_int_post = np.sum(new_post, axis=0)
        target.integrated_postcard = calculated_int_post # not part of final
        target.postcard = new_post # not part of final
        target.data_for_target(do_roll=True, ignore_bright=0)

        # print information about model

        new_uncerts = np.zeros_like(target.flux_uncert)
        new_uncerts[:] = target.flux_uncert

        ranges = []
        mins = []
        maxs = []
        avgs = []
        for i in range(models.shape[0]):
            curr_card = models[i]
            ranges.append(np.ptp(curr_card))
            mins.append(np.min(curr_card))
            maxs.append(np.max(curr_card))
            avgs.append(np.average(curr_card))

        print targ
        print np.ptp(ranges), np.std(ranges), np.var(ranges)
        print np.diff(ranges)
        print np.ptp(data_ranges), np.std(data_ranges), np.var(data_ranges)
        print np.diff(data_ranges)
        print np.ptp(mins), np.std(mins), np.var(mins)
        print np.diff(mins)
        print np.ptp(maxs), np.std(maxs), np.var(maxs)
        print np.diff(maxs)
        print np.ptp(avgs), np.std(avgs), np.var(avgs)
        print np.diff(avgs)
        print np.average(np.diff(ranges)), np.average(mins), np.average(maxs), np.average(avgs)
        print np.var(mins)/np.ptp(mins), np.var(maxs)/np.ptp(maxs), np.var(avgs)/np.ptp(avgs)

        bool_names = ["4x 20% of ran mins", "4x 20% of ran maxs", "4x 20% of ran avgs", "4x 20% of avg mins", "4x 20% of avg maxs", "4x 20% of avg avgs", "lower nanmean stds", "lower max stds    "]
        bool_res = [is_n_bools(np.abs(np.diff(mins)), 4, lambda x: x >= 0.2*np.ptp(mins)) \
                    , is_n_bools(np.abs(np.diff(maxs)), 4, lambda x: x >= 0.2*np.ptp(maxs)) \
                    , is_n_bools(np.abs(np.diff(avgs)), 4, lambda x: x >= 0.2*np.ptp(avgs)) \
                    , is_n_bools(np.abs(np.diff(mins)), 4, lambda x: x >= 0.2*np.average(mins)) \
                    , is_n_bools(np.abs(np.diff(maxs)), 3, lambda x: x >= 0.2*np.average(maxs)) \
                    , is_n_bools(np.abs(np.diff(avgs)), 3, lambda x: x >= 0.2*np.average(avgs)) \
                    , np.nanmean(new_uncerts) <= np.nanmean(old_uncerts) \
                    , is_std_better_biggest(old_uncerts, new_uncerts) \
        ]

        # TODO: also save integrated_post (new) + new_post as part of target attrs
        # TODO: check if should prodceed with model subtraction or not

        # if check_postcard_ranges(models):
        #     print targ, "TRUE BOII"
        #     integrated_post = np.sum(new_post, axis=0)
        #     target.data_for_target(do_roll=True, ignore_bright=0)
        # else:
        #     print targ, "FASLSESES"
        #     integrated_post = target.integrated_postcard


        # plot rest of stuff
        temp_int_post = np.zeros_like(calculated_int_post)
        temp_int_post[:] = calculated_int_post
        temp_int_post[min_i,min_j:max_j] = maximum
        temp_int_post[max_i,min_j:max_j] = maximum
        temp_int_post[min_i:max_i,min_j] = maximum
        temp_int_post[min_i:max_i,max_j] = maximum

        plt.figure(1)
        plt.subplot(1, 2, 2)
        plt.imshow(temp_int_post, interpolation='nearest', cmap='gray', vmin=min_img, vmax=max_img, origin='lower')
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

        return bool_names, bool_res

def calculate_accuracy(real_arr, res_arr):
    accurates = 0
    total = len(res_arr)
    for i, res in enumerate(res_arr):
        if res == real_arr[i]:
            accurates += 1
    return accurates/total

def format_arr(arr, sep="\t"):
    return sep.join(str(i) for i in arr)

def print_dict_res(kics, bool_names, all_res, real_res):
    print format_arr(["BOOLEAN NAMES:   "] + kics, "\t")

    for i, bool_name in enumerate(bool_names):
        formatted_arr = []
        curr_arr = []
        formatted_arr.append(bool_name)
        for j in range(len(kics)):
            formatted_arr.append(str(all_res[j][i]) + "\t")
            curr_arr.append(all_res[j][i])
        print format_arr(formatted_arr, "\t")
        print "\t\t\t\t%.3f" % calculate_accuracy(real_res, curr_arr)
        if curr_arr == real_res:
            print "\t HEY the results match!! for " +  str(kics[j])

    real_res = [str(x) + "\t" for x in real_res]
    print format_arr(["real results      "] + real_res)

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

    # kics1 = ["8527137", "8398294", "8397644", "8398286", "8398452", "10122937", "11873617", "3116513", "3116544", "3124279", "8381999"]
    # kics2 = ["8462852", "8115021", "8250547", "8250550", "8381999", "9091942"]
    # kics = (get_nth_kics(filename_stellar_params, 4000, 1, ' ', 0))[:]
    kics = ["11913365", "11913377"] + ben_kics
    # kics = ["11913365", "11913377"] + ["4555566", "4726114", "6708110", "8345997", "8759594", "9306271", "11415049", "11873617", "12417799"]

    all_res = []
    real_res = [True, True] + [False]*3 + [True]*2 + [False]*4 + [True] + [False]*6 + [True]*2 + [False] + [True] + [False]*4 + [True]*2 + [False]

    if len(real_res) != len(kics):
        print "real_res len needs to be same as kics len!"
        return 0

    for kic in kics:
        bool_names, bool_res = testing(kic, save_pdf=True)
        all_res.append(bool_res)

    print_dict_res(kics, bool_names, all_res, real_res)

    make_sound(0.3, 440)
    logger.info("### everything done ###")
    return 0

if __name__ == "__main__": # and __package__ is None:
    os.sys.path.append(f3_location)
    from f3 import photometry
    main()
