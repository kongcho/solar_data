"""
TODO:
- comments
"""

import os
import numpy as np
from settings import f3_location

from utils import clip_array
from settings import setup_logging

os.sys.path.append(f3_location)
from f3 import photometry

logger = setup_logging()

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

# helper function that determines 3-point pattern, but all quarters from the
#   same channel must have the same pattern
def is_more_or_less_all(target, quarters):
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

# helper function for is_more_or_less
# returns if all but one element is nonzero in an array
def most_are_same(arr):
    nos, counts = np.unique(arr, return_counts=True)
    for i, count in enumerate(counts):
        if count >= len(arr) - 1 and nos[i] != 0:
            return True, nos[i]
    return False, 0

# helper function that determines 3-point pattern
# all but one pattern from same channel must be the same
# returns -1 for downward, +1 for upward
def is_more_or_less(target, quarters):
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

# boolean function: determines if 3-point pattern exists
# aperture is too large (many stars in aperture) or too small (psf going out)
def is_large_ap(target):
    golden = range(0, 8)
    blacks = [[8, 9], [20, 21, 22], [31, 32, 33], [43, 44, 45]]
    reds = [[10, 11, 12], [23, 24, 25], [34, 35, 36], [46, 47, 48]]
    # blues: 13, 14 on same day, ignoring 13
    blues = [[14, 15, 16], [26, 27], [37, 38, 39], [49, 50, 51]]
    greens = [[17, 18, 19], [28, 29, 30], [40, 41, 42]]
    channels = [blacks, reds, blues, greens]
    for channel in channels:
        if is_more_or_less(target, channel) != 0:
            logger.info("is_large_ap True")
            return True
    logger.info("is_large_ap False")
    return False

# helper function for has_close_peaks: is peak if greater than all neighbours
#   and is brighter than center peak by factor, assumes center peak = target
def is_peak(max_of, xi0j0, xi0j1, xi0j2, xi1j0, xi2j0, factor=0.75):
    min_bright = factor * max_of
    others = [xi0j1, xi0j2, xi1j0, xi2j0]
    booleans = [not (xi0j0 == 0 and all(x == 0 for x in others))
                , all(x < xi0j0 for x in others)
                , xi0j0 >= min_bright
               ]
    return all(booleans)

# boolean function: determines if there's a peak within a given distance
#   around center point (target star)
def has_close_peaks(target, diff=7, min_factor=1, avoid_pixels=0):
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
                logger.info("has_close_peaks True")
    if len(peaks) == 0:
        logger.info("has_close_peaks False")
    return any(peaks)

# boolean function: poor estimate for faint stars based on aperture
def is_faint_rough(target, limit=5500000):
    c_i = (img.shape[0])//2
    c_j = (img.shape[1])//2
    is_faint = True
    for i in range(c_i - 1, c_i + 2):
        for j in range(c_i - 1, c_i + 2):
            if target.img[i][j] != 0.0 and target.img[i][j] >= limit:
                is_faint = False
                break
    logger.info("is_faint done: %s" % is_faint)
    return is_faint

# boolean function: always passes every star, for testing
def fake_bool(target):
    logger.info("fake_bool done")
    return True

# TODO: helper function
def pad_img(img, desired_shape, positions, pad_val=0):
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

# TODO: function
def pad_img_wrap(img, desired_shape, sides, pad_val=0):
    offset_x = 0
    offset_y = 0
    if "Top" in sides:
        offset_y += desired_shape[0]-img.shape[0]
    if "Left" in sides:
        offset_x += desired_shape[1]-img.shape[1]
    return pad_img(img, desired_shape, (offset_y, offset_x), pad_val)

def is_n_bools(arr, n, bool_func):
    n_bools = False
    for i in arr:
        if bool_func(i):
            n -= 1
        if n == 0:
            n_bools = True
            break
    return n_bools

def is_second_star(img, xi0j0, xi0j1, xi0j2, xi1j0, xi2j0, factor=0.75):
    min_bright = factor * (np.max(img)) #- np.min(img[np.nonzero(img)]))
    others = [xi0j1, xi0j2, xi1j0, xi2j0]
    booleans = [any(others)
                , is_n_bools(others, 1, lambda x: x == 0)
                , all(x < xi0j0 for x in others)
                , xi0j0 >= min_bright
                ]
    return all(booleans)

def remove_second_star(img, min_factor):
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

def monotonic_arr(arr, is_decreasing, relax_pixels=2, diff_flux=0):
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

def img_to_new_aperture(target, img, image_region=15):
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
    return img

def isolate_star_cycle(img, ii, jj, image_region=15, relax_pixels=2):
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

def improve_aperture(target, mask=None, image_region=15, relax_pixels=2):
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
            img_cycle = isolate_star_cycle(img_cycle, ii, jj, image_region, 1)
        else:
            img_cycle = isolate_star_cycle(img_cycle, ii, jj, image_region, relax_pixels)
        run_cycle = np.any(np.subtract(img_save, img_cycle))
        img_save = img_cycle

    remove_second_star(img_save, 0.7)

    img_to_new_aperture(target, img_save, image_region)
    target.data_for_target(do_roll=True, ignore_bright=0)

    logger.info("improve_aperture done")
    return target.img

def calculate_better_aperture(target, mask_factor=0.001, image_region=15):
    tar = target.target
    channel = [tar.params['Channel_0'], tar.params['Channel_1'],
               tar.params['Channel_2'], tar.params['Channel_3']]
    kepprf = lk.KeplerPRF(channel=channel[0], shape=(image_region*2, image_region*2), \
                          column=image_region, row=image_region)
    prf = kepprf(flux=1000, center_col=image_region*2, center_row=image_region*2, \
                 scale_row=1, scale_col=1, rotation_angle=0)
    mask = np.where(prf > mask_factor*np.max(prf), 1, 0)
    improve_aperture(target, mask, image_region)
    return target

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

