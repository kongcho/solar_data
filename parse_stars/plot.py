# TODO: FIX IT UP

import logging
import os
import numpy as np

import matplotlib
if os.environ.get('DISPLAY','') == "":
    print("Using non-interactive Agg backend")
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from matplotlib.backends.backend_pdf import PdfPages

from utils import get_sub_kwargs
from aperture import run_photometry, improve_aperture, calculate_better_aperture
from settings import setup_logging

logger = setup_logging()

# creates plot for one target, assumes already have obs_flux, flux_uncert
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

# helper function for plot_targets that sets up photometry of a star
#   runs a photometry and tests a list of boolean functions on it
#   then creates a plot for it with plot_data
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

# plots list of targets to a filename if the boolean function is true
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

# plots light curves after each given function
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

def plot_box(x1, x2, y1, y2, marker='r-', **kwargs):
    plt.plot([x1, x1], [y1, y2], marker, **kwargs)
    plt.plot([x2, x2], [y1, y2], marker, **kwargs)
    plt.plot([x1, x2], [y1, y1], marker, **kwargs)
    plt.plot([x1, x2], [y2, y2], marker, **kwargs)

def plot_background_modelling(targ, fout="./", image_region=15, model_pix=15, mask_factor=0.001, \
                              max_factor=0.2, min_img=-1000, max_img=1000, save_pdf=True):
    target = run_photometry(target)
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

def is_std_better_biggest(old_stds, stds):
    max_i = np.argmax(stds)
    return stds[max_i] <= old_stds[max_i]

def is_std_better_avg(old_stds, stds):
    return np.nanmean(stds) <= np.nanmean(old_stds)

# runs through lists of different parameters, print flux plots and apertures,
#   then tests if it has only 1 peak, then takes out lowest stddev
def print_better_apertures(targ, boolean_func, edge_lim=0.015, min_val=5000, \
                           extend_region_size=3, remove_excess=4):

    target = photometry.star(targ)
    target.make_postcard()

    edge_lims = np.arange(edge_lim - 0.010, edge_lim + 0.025, 0.005)
    min_vals = np.arange(min_val - 2000, min_val + 2000, 500)
    region_sizes = np.arange(2, 5)
    excesses = np.arange(2, 6)

    test_vars = [edge_lims, min_vals, region_sizes, excesses]
    vals = [list(itertools.product(*test_vars))]

    run_partial_photometry(target, edge_lim=0.015, min_val=5000, extend_region_size=3, \
                           remove_excess=4, ntargets=100)

    old_stds = target.flux_uncert
    plot_data(target)

    with PdfPages(output_file) as pdf:
        for count, val in enumerate(vals, 1):
            res = {}
            run_partial_photometry(target, edge_lim=val[0], min_val=val[1], \
                                   extend_region_size=val[2], remove_excess=val[3], ntargets=100)
            res["settings"] = "edge: " + str(val[0]) + " min: " + str(val[1]) + \
                              " region: " + str(val[2]) + " excess: " + str(val[3])
            res["boolean"] = boolean_func(target)
            res["is_avg"] = is_std_better_avg(old_stds, target.flux_uncert)
            res["is_most"] = is_std_better_biggest(old_stds, target.flux_uncert)
            res["has_peaks"] = has_close_peaks(target)
            results[val] = res
            plot_data(target, count)
            plt.gcf().text(4/8.5, 1/11., str(res), ha='center', fontsize = 11)
            pdf.savefig()
            plt.close()
    logger.info("done")
    return

# TODO: function
def print_best_apertures(targ, edge_lim=0.015, min_val=5000, extend_region_size=3, \
                         remove_excess=4, min_factor=0.7):
    fout = targ + "_plot.pdf"
    target = photometry.star(targ, ffi_dir=ffidata_folder)
    target.make_postcard()

    best_pars = (edge_lim, min_val, extend_region_size, remove_excess)
    edge_lims = np.arange(edge_lim - 0.010, edge_lim + 0.025, 0.01)
    min_vals = np.arange(min_val - 2000, min_val + 2000, 1000)
    region_sizes = np.arange(1, 3)
    excesses = np.arange(1, 4)

    single_results = []
    all_vals = []

    for v0, v1, v2, v3 in itertools.product(edge_lims, min_vals, region_sizes, excesses):
        all_vals.append((v0, v1, v2, v3))

    with PdfPages(targ + "_plot_1.pdf") as pdf, PdfPages(targ + "_plot_2.pdf") as pdf2:
        if run_partial_photometry(target, edge_lim=0.015, min_val=5000, \
                              extend_region_size=3, remove_excess=4, ntargets=100) == 1:
            return 1

        best_unc = target.flux_uncert
        plot_data(target)
        plt.gcf().text(4/8.5, 1/11., str(best_pars), ha='center', fontsize = 11)
        pdf.savefig()
        pdf2.savefig()
        plt.close()

        for count, vals in enumerate(all_vals, 1):
            res = {}
            if run_partial_photometry(target, edge_lim=vals[0], min_val=vals[1], \
                                      extend_region_size=vals[2], remove_excess=vals[3], \
                                      ntargets=100) == 1:
                continue
            res["settings"] = vals
            res["has_peaks"] = has_peaks(target, min_factor)
            res["is_avg"] = is_std_better_avg(best_unc, target.flux_uncert)
            res["is_most"] = is_std_better_biggest(best_unc, target.flux_uncert)
            plot_data(target)
            plt.gcf().text(4/8.5, 1/11., str(res), ha='center', fontsize = 11)
            pdf2.savefig()
            plt.close()
            if not res["has_peaks"]:
                single_results.append((np.nanmean(target.flux_uncert), vals))

        if len(single_results) != 0:
            best_unc, best_pars = single_results[single_results.index(min(single_results))]

        if run_partial_photometry(target, edge_lim=best_pars[0], min_val=best_pars[1], \
                                  extend_region_size=best_pars[2], remove_excess=best_pars[3], \
                                  ntargets=100) == 1:
            return 1
        plot_data(target)
        plt.gcf().text(4/8.5, 1/11., str((best_pars, best_unc)), ha='center', fontsize = 11)
        pdf.savefig()
        pdf2.savefig()
        plt.close()
    logger.info("done")
    return 0
