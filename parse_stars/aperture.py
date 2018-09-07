from utils import clip_array, is_n_bools
from settings import setup_logging, ffidata_folder, f3_location
logger = setup_logging()

import os
import numpy as np
import lightkurve as lk

os.sys.path.append(f3_location)
from f3 import photometry


def pad_img(img, desired_shape, positions, pad_val=0):
    """
    helper
    pads any image to desired shape, filled with pad_val

    :img: current image to pad
    :desired_shape: shape to pad to
    :positions: position of the given image within an empty desired shape
    :pad_val: value to pad image with

    :return: new image with desired shape
    """
    if len(desired_shape) != len(positions):
        logger.error("pad_img: odd required shape dimensions")
        return img
    pads = []
    for i, position in enumerate(positions):
        pad_len = desired_shape[i] - position - img.shape[i]
        if pad_len < 0 or position < 0:
            logger.error("pad_img: invalid positions")
            return img
        pads.append((position, pad_len))
    return np.pad(img, pads, mode='constant', constant_values=pad_val)


# function, pads image to fit desired_shape and fill up extra space with pad_val
def pad_img_wrap(img, desired_shape, sides, pad_val=0):
    """
    wrapper that pads image to fit desired_shape and fill up extra space with pad_val
    determines where the current image position is based on which side it is on

    :img: current image to pad
    :desired_shape: shape to pad to
    :sides: which side of the image needs to be padded
    :pad_val: value to pad image with

    :return: new image with desired shape
    """
    offset_x = 0
    offset_y = 0
    if "Top" in sides:
        offset_y += desired_shape[0]-img.shape[0]
    if "Left" in sides:
        offset_x += desired_shape[1]-img.shape[1]
    img = pad_img(img, desired_shape, (offset_y, offset_x), pad_val)
    logger.info("done")
    return img


def run_partial_photometry(target, image_region=15, edge_lim=0.015, min_val=5000, ntargets=100, \
                           extend_region_size=3, remove_excess=4, plot_window=15, plot_flag=False):
    """
    helper
    assumes already has a target class setup for the kic
    runs photometry to get the light curves and aperture based on given settings (see f3)
    """

    try:
        target.find_other_sources(edge_lim, min_val, ntargets, extend_region_size, \
                                  remove_excess, plot_flag, plot_window)
    except Exception as e:
        logger.info("unsuccessful: %s" % target.kic)
        logger.error(e, exc_info=True)
        return 1

    target.data_for_target(do_roll=True, ignore_bright=0)

    ii, jj = target.center
    ii, jj = int(ii), int(jj)

    sides = []
    if ii-image_region < 0:
        sides.append("Top")
    if jj-image_region < 0:
        sides.append("Left")

    ymin = max(ii-image_region, 0)
    ymax = min(ii+image_region, target.postcard.shape[1])
    xmin = max(jj-image_region, 0)
    xmax = min(jj+image_region, target.postcard.shape[2])

    targets_small = target.targets[ymin:ymax, xmin:xmax]
    targets_pad = pad_img_wrap(targets_small, (image_region*2, image_region*2), sides, 0.0)

    int_small = target.integrated_postcard[ymin:ymax, xmin:xmax]
    int_pad = pad_img_wrap(int_small, (image_region*2, image_region*2), sides, 0.0)

    img = (targets_pad == 1)*(int_pad + target.postcard.shape[0]*100000)

    setattr(photometry.star, 'img', img)

    logger.info("done: %s" % target.kic)
    return target


def run_photometry(targ, image_region=15, edge_lim=0.015, min_val=5000, ntargets=100, \
                   extend_region_size=3, remove_excess=4, plot_window=15, plot_flag=False):
    """
    creates target class and runs photometry on it, gets aperture and light curve

    :return: target
    """
    try:
        target = photometry.star(targ, ffi_dir=ffidata_folder)
        target.make_postcard()
    except Exception as e:
        logger.info("unsuccessful: %s" % targ)
        logger.error(e.message)
        return 1

    return run_partial_photometry(target, image_region, edge_lim, min_val, ntargets, \
                                  extend_region_size, remove_excess, plot_window, plot_flag)


def is_more_or_less_all(target, quarters):
    """
    examines light curve boolean function
    sees if light curve has a 3-point pattern for all quarters within the same channel
    """
    is_strange = 0
    is_incr = False
    is_decr = False
    for indexes in quarters:
        for i in range(len(indexes) - 1):
            if target.obs_flux[indexes[i + 1]] > target.obs_flux[indexes[i]]:
                if is_decr:
                    is_decr = False
                    break
                is_incr = True
            elif target.obs_flux[indexes[i + 1]] < target.obs_flux[indexes[i]]:
                if is_incr:
                    is_incr = False
                    break
                is_decr = True
        if is_strange == 0:
            if is_incr:
                is_strange = 1
            elif is_decr:
                is_strange = -1
            else:
                return 0
        elif (is_strange == 1 and not is_incr) or (is_strange == -1 and not is_decr):
            return 0
    return is_strange


def most_are_same(arr):
    """
    helper for is_more_or_less
    returns if all but one element is nonzero in an array

    :arr: array of -1 (downward), +1 (upward) pattern for given channel

    :return: boolean if most are the same, number of times they are the same
    """
    nos, counts = np.unique(arr, return_counts=True)
    for i, count in enumerate(counts):
        if count >= len(arr) - 1 and nos[i] != 0:
            return True, nos[i]
    return False, 0


def is_more_or_less(target, quarters):
    """
    helper function that determines 3-point pattern
    checks that all but one pattern from same channel must be the same

    :return: 0 if no pattern, -1 if downward pattern, +1 if upward pattern
    """
    curr_ch = []
    is_incr = False
    is_decr = False
    for indexes in quarters:
        for i in range(len(indexes) - 1):
            if target.obs_flux[indexes[i + 1]] > target.obs_flux[indexes[i]]:
                if is_decr:
                    is_decr = False
                    break
                is_incr = True
            elif target.obs_flux[indexes[i + 1]] < target.obs_flux[indexes[i]]:
                if is_incr:
                    is_incr = False
                    break
                is_decr = True
        is_strange = 1 if is_incr else -1 if is_decr else 0
        curr_ch.append(is_strange)
    is_nontrivial = most_are_same(curr_ch)
    if is_nontrivial:
        return is_nontrivial[1]
    return 0


def is_large_ap(target):
    """
    examines light curve boolean function: if 3-point pattern exists
    implies aperture is too large (many stars in aperture) or too small (psf going out)

    :return: boolean
    """
    golden = range(0, 8)
    blacks = [[8, 9], [20, 21, 22], [31, 32, 33], [43, 44, 45]]
    reds = [[10, 11, 12], [23, 24, 25], [34, 35, 36], [46, 47, 48]]
    # blues: 13, 14 on same day, ignoring 13
    blues = [[14, 15, 16], [26, 27], [37, 38, 39], [49, 50, 51]]
    greens = [[17, 18, 19], [28, 29, 30], [40, 41, 42]]
    channels = [blacks, reds, blues, greens]
    for channel in channels:
        if is_more_or_less(target, channel) != 0:
            logger.info("True")
            return True
    logger.info("False")
    return False


def is_peak(max_of, xi0j0, xi0j1, xi0j2, xi1j0, xi2j0, factor=0.75):
    """
    helper for has_close_peaks
    a peak isn't surrounded by only 0.0s, is greater than neighbours,
      and is brighter than factor of center peak

    :max_of: value multiplied by factor as threshold
    :xi0j0: center pixel
    :xi0j1, xi0j2, xi1j0, xi2j0: surrounding pixels
    :factor: multiplied by max_of as threshold to determine if is a peak

    :return: boolean
    """
    min_bright = factor * max_of
    others = [xi0j1, xi0j2, xi1j0, xi2j0]
    booleans = [not (xi0j0 == 0 and all(x == 0 for x in others))
                , all(x < xi0j0 for x in others)
                , xi0j0 >= min_bright
               ]
    return all(booleans)


def has_close_peaks(target, diff=7, min_factor=1, avoid_pixels=0):
    """
    examines light curve boolean function: if there is a peak around distance of center point

    :diff: distance from center of aperture
    :min_factor: factor of center pixel value to determine threshold for a peak or not
    :avoid_pixels: distance from center of aperture to be immune to be a peak

    :return: boolean
    """
    peaks = []
    img = target.img
    len_x = img.shape[1]
    len_y = img.shape[0]
    c_i = len_y//2
    c_j = len_x//2
    if diff is None or c_i <= diff or diff <= 0:
        min_i = 1
        max_i = len_y - 1
    else:
        min_i = c_i - diff
        max_i = c_i + diff
    if diff is None or c_j <= diff or diff <= 0:
        min_j = 1
        max_j = len_x - 1
    else:
        min_j = c_j - diff
        max_j = c_j + diff
    avoids_i = range(c_i-avoid_pixels, c_i+1+avoid_pixels)
    avoids_j = range(c_j-avoid_pixels, c_j+1+avoid_pixels)
    for i in range(min_i, max_i):
        for j in range(min_j, max_j):
            if i in avoids_i and j in avoids_j:
                continue
            if is_peak(img[c_i, c_j], img[i, j], img[i, j-1], img[i, j+1], \
                       img[i-1, j], img[i+1, j], min_factor):
                peaks.append((i, j))
                logger.info("True")
    if len(peaks) == 0:
        logger.info("False")
    return any(peaks)


def get_boolean_stars(targets, boolean_funcs, edge_lim=0.015, min_val=500, ntargets=100):
    """
    output dictionary of functions and list of kics that are True for those functions

    :targets: list of target class instances
    :boolean_funcs: list of boolean functions
    :edge_lim, min_val, ntargets: see run_photometry / f3

    :return: dictionary
    """
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
    logger.info("done")
    return full_dict


def is_second_star(img, xi0j0, xi0j1, xi0j2, xi1j0, xi2j0, factor=0.75):
    """
    helper, determines how the second star is detected
    a second star is at the edge of detector, is greated than its neighbours and threshold

    :img: aperture image
    :xi0j0: center pixel
    :xi0j1, xi0j2, xi1j0, xi2j0: surrounding pixels
    :factor: multiplied by image maximum as threshold

    :return: boolean
    """
    min_bright = factor * (np.max(img))
    others = [xi0j1, xi0j2, xi1j0, xi2j0]
    booleans = [any(others)
                , is_n_bools(others, 1, lambda x: x == 0)
                , all(x < xi0j0 for x in others)
                , xi0j0 >= min_bright
                ]
    return all(booleans)

def remove_second_star(img, min_factor):
    """
    helper for improve_aperture, removes any detected second stars by pixel
    """
    removes = []
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if is_second_star(img, img[i, j], img[i, j-1], img[i, j+1], \
                              img[i-1, j], img[i+1, j], min_factor):
                removes.append((i, j))
    for coord in removes:
        i, j = coord
        img[i, j] = 0
    return img


def img_to_new_aperture(target, img, image_region=15):
    """
    modifies new aperture into target information to calculate new light curves

    :target: target instance
    :img: new aperture
    :image_region: distance from center pixel to capture for aperture
    """
    target.img = img
    ii, jj = target.center
    ii, jj = int(ii), int(jj)
    len_y = img.shape[0]
    len_x = img.shape[1]

    for i in range(len_y):
        for j in range(len_x):
            if img[i, j] == 0:
                big_i, big_j = clip_array([i+ii-image_region, j+jj-image_region], \
                                          [target.targets.shape[0]-1, target.targets.shape[1]-1], \
                                          [True, True])
                target.targets[big_i, big_j] = 0
    logger.info("done")
    return img


def monotonic_arr(arr, is_decreasing, relax_pixels=2, diff_flux=0):
    """
    helper for improve_aperture
    returns index where array decreases/increases when it's supposed to increase/decrease

    :arr: 1D array to process
    :is_decreasing: boolean if is supposed to increase/decrease
    :relax_pixels: number of pixels that the pattern has to break to count
    :diff_flux: minimum difference between consecutive pixels to be in one direction

    :return: index of where the pattern breaks,
      0 if whole array increases, -1 if whole array decreases
    """
    new_arr = arr if is_decreasing else np.flip(arr, 0)
    diffs = np.diff(new_arr)
    length = range(len(diffs))
    lengths = []
    for start_i in range(relax_pixels):
        lengths.append(length[start_i:])
    for i, diff in enumerate(zip(*lengths)):
        if all(diffs[list(diff)] > diff_flux) and not (arr[i] == 0 and all(arr[[i-1, i+1]] != 0)):
            return diff[0] if is_decreasing else len(arr)-(diff[-1]+1)
    return -1 if is_decreasing else 0


def isolate_star_cycle(img, ii, jj, relax_pixels=2, image_region=15):
    """
    helper for improve_aperture
    removes potential stars by assuming pixels have decreasing flux away from the star

    :img: aperture
    :ii, jj: center of aperture
    :relax_pixels: how many pixels are required to break the pattern (how relaxed it is)
    :image_region: distance around the center of the star for aperture

    :return: new aperture
    """
    len_x = img.shape[1]
    len_y = img.shape[0]
    c_j = len_x//2
    c_i = len_y//2

    # go through rows
    for i in range(len_y):
        targets_i = i+ii-image_region
        inc = monotonic_arr(img[i,:c_j], False, relax_pixels, 0)
        if inc != 0:
            img[i,:inc] = 0
        dec = monotonic_arr(img[i,(c_j+1):], True, relax_pixels, 0)
        if dec != -1:
            real_dec = c_j + 1 + dec
            img[i,real_dec:] = 0

    # go through cols
    for j in range(len_x):
        targets_j = j+jj-image_region
        inc = monotonic_arr(img[:c_i,j], False, relax_pixels, 0)
        if inc != 0:
            img[:inc,j] = 0
        dec = monotonic_arr(img[(c_i+1):,j], True, relax_pixels, 0)
        if dec != -1:
            real_dec = c_i + 1 + dec
            img[real_dec:,j]=0

    return img


def improve_aperture(target, mask=None, relax_pixels=2, second_factor=0.7, image_region=15):
    """
    improves aperture by removing other stars from initial aperture

    restricts aperture to given mask first, then removes other stars with strict parameter
      if they are detected, relaxed parameters if not, then removes remaining second stars
      at aperture edge

    :mask: mask to restrict aperture to
    :relax_pixels: how relaxed the function is if doesn't detect stars
    :second_factor: factor threshold for second star removal
    :image_region: distance from central pixel for aperture

    :return: new aperture
    """
    ii, jj = target.center
    ii, jj = int(ii), int(jj)

    if not np.any(target.img):
        return target

    run_cycle = True
    img_save = np.empty_like(target.img)
    img_save[:] = target.img

    if mask.shape == img_save.shape and (mask is not None or not np.any(mask)):
        img_save = np.multiply(mask, target.img)

    while run_cycle:
        img_cycle = np.empty_like(img_save)
        img_cycle[:] = img_save
        if has_close_peaks(target, None, 1.2):
            img_cycle = isolate_star_cycle(img_cycle, ii, jj, 1, image_region)
        else:
            img_cycle = isolate_star_cycle(img_cycle, ii, jj, relax_pixels, image_region)
        run_cycle = np.any(np.subtract(img_save, img_cycle))
        img_save = img_cycle

    remove_second_star(img_save, second_factor)

    img_to_new_aperture(target, img_save, image_region)
    target.data_for_target(do_roll=True, ignore_bright=0)

    logger.info("done")
    return target.img


def calculate_aperture_mask(target, mask_factor=0.001, image_region=15):
    """
    helper that creates mask to overlay aperture with from rough PSF from lightkurve package

    :mask_factor: factor multiplied maximum of given PSF from lightkurve
    :image_region: distance from center of pixel as part of aperture

    :return: mask (1s, 0s) as image, all 1s if no channel available
    """
    tar = target.target
    channel = [tar.params['Channel_0'], tar.params['Channel_1'],
               tar.params['Channel_2'], tar.params['Channel_3']]
    for chan in channel:
        if chan is not None:
            first_channel = chan
            break
    else:
        return np.ones((image_region*2, image_region*2))
    kepprf = lk.KeplerPRF(channel=first_channel, shape=(image_region*2, image_region*2), \
                          column=image_region, row=image_region)
    prf = kepprf(flux=1000, center_col=image_region*2, center_row=image_region*2, \
                 scale_row=1, scale_col=1, rotation_angle=0)
    mask = np.where(prf > mask_factor*np.max(prf), 1, 0)
    return mask


def calculate_better_aperture(target, mask_factor=0.001, relax_pixels=2, \
                              second_factor=0.7, image_region=15):
    """
    combines creating a mask and improving aperture

    :mask_factor, relax_pixels, second_factor, image_region: see above
    """
    mask = calculate_aperture_mask(target, mask_factor, image_region)
    improve_aperture(target, mask, relax_pixels, second_factor, image_region)
    logger.info("done")
    return 0


def logical_or_all_args(*args):
    """
    helper, completes logical arr on given arrays, all arrays should be same shape
    """
    result = np.zeros_like(args[0])
    for arg in args:
        result += arg
    return np.where(result != 0, 1, 0)


def make_background_mask(target, img, coords, max_factor=0.2):
    """
    helper that creates mask for the other stars in the background

    :target: target instance with given kic
    :img: aperture
    :coords: region edge coordinates to calculate mask from
    :max_factor: float, percentile to get max_factor% greatest pixels as a mask

    :return: mask for background stars
    """
    if not np.any(img):
        return -1

    min_i, max_i, min_j, max_j = coords
    max_mask = np.where(img >= np.percentile(img, int((1-max_factor)*100)), 1, 0)
    targets_mask = np.where(target.targets != 0, 1, 0)[min_i:max_i, min_j:max_j]
    mask = logical_or_all_args(max_mask, targets_mask)
    return mask


def model_background(target, max_factor=0.2, model_pix=15):
    """
    models the background variation and removes it to calculate the light curves
    models background as median of masked background

    :target: target instance
    :max_factor: see make_background_mask
    :model_pix: distance from center pixel to model with (is not img_region)

    :return: target instance
    """
    coords = clip_array([target.center[0]-model_pix, target.center[0]+model_pix, \
                         target.center[1]-model_pix, target.center[1]+model_pix], \
                        [0, target.postcard.shape[1]-1, 0, target.postcard.shape[2]-1], \
                        [False, True, False, True])
    min_i, max_i, min_j, max_j = coords

    for i in range(target.postcard.shape[0]):
        min_i, max_i, min_j, max_j = coords
        region = target.postcard[i]
        img = region[min_i:max_i, min_j:max_j]

        mask = make_background_mask(target, img, coords, max_factor)
        z = np.ma.masked_array(img, mask=mask)
        img -= np.ma.median(z)

    target.integrated_postcard = np.sum(target.postcard, axis=0)
    target.data_for_target(do_roll=True, ignore_bright=0)
    logger.info("done")
    return target


def get_aperture_center(target, image_region=15):
    """
    gets aperture center location if the star is within image_region of edge of postcard
    """
    aperture_center = [image_region, image_region]
    for i in range(2):
        center_i = int(target.center[i])
        if center_i < image_region:
            aperture_center[i] = center_i
    return aperture_center
