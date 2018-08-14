from utils import get_sub_kwargs, clip_array, build_arr_n_names, format_arr
from aperture import run_photometry, improve_aperture, get_aperture_center, \
    calculate_better_aperture, model_background, make_background_mask
from settings import setup_logging, mpl_setup

logger = setup_logging()
mpl_setup()

import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from matplotlib.backends.backend_pdf import PdfPages


# function, creates plot for one target, assumes already have obs_flux, flux_uncert
def plot_data(target, count=0):
    fig = plt.figure(figsize=(11,8))
    gs.GridSpec(3,3)

    plt.subplot2grid((3,3), (1,2))
    plt.title(target.kic, fontsize=20)
    plt.imshow(target.img, interpolation='nearest', cmap='gray', vmin=98000*52, vmax=104000*52)

    plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
    for i in range(4):
        g = np.where(target.qs == i)[0]
        plt.errorbar(target.times[g], target.obs_flux[g], \
                     yerr=target.flux_uncert[i], fmt=target.fmt[i])
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Relative Flux', fontsize=15)

    fig.text(7.5/8.5, 0.5/11., str(count + 1), ha='center', fontsize = 12)
    fig.tight_layout()

    logger.info("done")
    return fig


# helper for plot_targets that sets up photometry of a star
# runs a photometry and tests a list of boolean functions on it then creates a plot for it
def tests_booleans(targ, boolean_funcs, count, pick_bad=True, edge_lim=0.015, \
                   min_val=5000, ntargets=100):
    target = run_photometry(targ, edge_lim=edge_lim, min_val=min_val, ntargets=ntargets)
    if target == 1:
        return target

    for boolean in boolean_funcs:
        if pick_bad and not boolean(target):
            return 1
        elif boolean(target):
            return 1
    plot_data(target, count)
    logger.info("done")
    return target


# function, plots list of targets to a filename if the boolean function is true
def plot_targets(filename, boolean_funcs, targets, pick_bad=True):
    filename = filename.rsplit(".", 1)[0]
    total = len(targets)
    count = 1
    parsed_targets = []
    if len(boolean_funcs) == 1:
        output_file = filename + "_" + str(boolean_funcs[0].__name__) + ".pdf"
    else:
        output_file = filename + "_bads.pdf"
    with PdfPages(output_file) as pdf:
        for targ in targets:
            logger.info("# " + targ)
            target = tests_booleans(targ, boolean_funcs, count, pick_bad)
            if target != 1:
                plt.gcf().text(4/8.5, 1/11., "" , ha='center', fontsize = 12)
                parsed_targets.append(target)
                pdf.savefig()
                plt.close()
                logger.info(str(count) + "\t" + targ + "\tplot_done")
                count += 1
    logger.info(str(count - 1) + " out of " + str(total) + " targets plotted")
    logger.info("done")
    return parsed_targets


# function, plots light curves after each given function
# kwargs is any optional argument for any function
def plot_functions(targ, fout="./", save_fig=True, *funcs, **kwargs):
    target = run_photometry(targ)
    if target == 1:
        return target

    with PdfPages(fout + targ + "_out.pdf") as pdf:
        for i in range(len(funcs) + 1):
            if i != 0:
                func = funcs[i-1]
                curr_kwargs = get_sub_kwargs(func, **kwargs)
                func(target, **curr_kwargs)
            fig = plot_data(target)
            if save_fig:
                plt.gcf().text(4/8.5, 1/11., str(np.nanmean(target.flux_uncert)), \
                               ha='center', fontsize = 11)
                pdf.savefig()
                plt.close(fig)

    if not save_fig:
        plt.show()
    plt.close("all")

    logger.info("done")
    return target


# helper, plots a box for given coordinates
def plot_box(x1, x2, y1, y2, marker='r-', **kwargs):
    plt.plot([x1, x1], [y1, y2], marker, **kwargs)
    plt.plot([x2, x2], [y1, y2], marker, **kwargs)
    plt.plot([x1, x2], [y1, y1], marker, **kwargs)
    plt.plot([x1, x2], [y2, y2], marker, **kwargs)


# function, plots the different light curves for original calculation, with improved aperture,
#   and the effects of background modelling
def plot_background_modelling(targ, fout="./", image_region=15, model_pix=15, mask_factor=0.001, \
                              max_factor=0.2, min_img=-1000, max_img=1000, save_pdf=True):
    target = run_photometry(targ)
    if target == 1:
        return target

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

    with PdfPages(fout + targ + "_out.pdf") as pdf:
        # original plot
        fig0 = plot_data(target)
        plt.gcf().text(4/8.5, 1/11., str(np.nanmean(target.flux_uncert)), \
                       ha='center', fontsize = 11)
        pdf.savefig()
        plt.close(fig0)

        # improved aperture plot
        calculate_better_aperture(target, mask_factor=mask_factor, image_region=image_region)
        fig0 = plot_data(target)
        plt.gcf().text(4/8.5, 1/11., str(np.nanmean(target.flux_uncert)), \
                       ha='center', fontsize = 11)
        pdf.savefig()
        plt.close(fig0)

        # background modelling
        for i in range(target.postcard.shape[0]):
            # make model
            region = target.postcard[i]
            img = region[min_i:max_i, min_j:max_j]

            mask = make_background_mask(target, img, coords, max_factor, model_pix)
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

    logger.info("done")
    return target


# function, prints aperture, calculated light curves before and after improvement to 3 different text files
def print_lc_improved(kics, fouts, min_distance=20, print_headers=True):
    fout0, fout1, fout2 = fouts
    successes = 0
    is_first = True

    if print_headers:
        names0 = ["KIC"] + build_arr_n_names("postcard_center", 2) + \
                 build_arr_n_names("aperture_center", 2) + build_arr_n_names("img", 900) + ["Note"]
        names1 = ["KIC"] + build_arr_n_names("flux_old", 52) + \
                 build_arr_n_names("est_unc_old", 4) + build_arr_n_names("mod_unc_old", 52)
        names2 = ["KIC"] + build_arr_n_names("flux_new", 52) + \
                 build_arr_n_names("est_unc_new", 4) + build_arr_n_names("mod_unc_new", 52)

    with open(fout0, "ab") as f0, open(fout1, "ab") as f1, open(fout2, "ab") as f2:
        w0 = csv.writer(f0, delimiter=',', lineterminator='\n')
        w1 = csv.writer(f1, delimiter=',', lineterminator='\n')
        w2 = csv.writer(f2, delimiter=',', lineterminator='\n')

        for kic in kics:
            target = run_photometry(kic)
            if target == 1:
                continue
            if print_headers and is_first:
                names1 = ["KIC"] + map(str, target.times) + build_arr_n_names("est_unc_old", 4) + \
                         build_arr_n_names("mod_unc_old", 52)
                names2 = ["KIC"] + map(str, target.times) + build_arr_n_names("uncert_new", 4) + \
                         build_arr_n_names("mod_unc_new", 52)
                w0.writerow(names0)
                w1.writerow(names1)
                w2.writerow(names2)
                is_first = False

            targ = target.target
            col = [targ.params['Column_0'], targ.params['Column_1'],
                   targ.params['Column_2'], targ.params['Column_3']]
            row = [targ.params['Row_0'], targ.params['Row_1'],
                   targ.params['Row_2'], targ.params['Row_3']]
            center = [str(row), str(col)]
            if np.nanmin(col) <= min_distance or np.nanmin(row) <= min_distance:
                flag = "Close to edge"
            else:
                flag = "--"

            target.model_uncert()
            calculate_better_aperture(target, 0.001, 2, 0.7, 15)
            arr0 = np.concatenate([np.asarray([kic]), center, get_aperture_center(target), \
                                   target.img.flatten(), np.asarray([flag])])
            arr1 = np.concatenate([np.asarray([kic]), target.obs_flux, \
                                   target.flux_uncert, target.target_uncert])
            model_background(target, 0.2, 15)
            arr2 = np.concatenate([np.asarray([kic]), target.obs_flux, \
                                   target.flux_uncert, target.target_uncert])

            w0.writerow(arr0)
            w1.writerow(arr1)
            w2.writerow(arr2)

            successes += 1
            logger.info("done: %s" % kic)
    logger.info("done: %d kics", successes)
    return 0
