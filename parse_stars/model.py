"""
FUNCTIONS THAT DETERMINES A STAR'S VARIABILITY FROM LIGHT CURVE
"""

from aperture import run_photometry

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
from scipy import optimize

def damped_sine_model(x, inits):
    return inits[0]*np.exp(-1*inits[1]*x)*np.sin(2*np.pi*inits[2]*x + inits[3])

def sine_model(x, inits):
    return inits[0]*np.sin(2*np.pi*inits[1]*x + inits[2])

def log_model(x, inits):
    return inits[0] + inits[1]*np.log(x)

def simple_err_func(inits, x, y, model_func):
    return model_func(x, inits) - y

def make_scipy_model(x, y, label, model_func, err_func, init_arr):
    p, success = optimize.leastsq(err_func, init_arr[:], args=(x, y, model_func))
    model = model_func(x, p)
    return model, label

def make_astropy_model(x, y, label, model_func, fitting, *args, **kwargs):
    init = model_func(*args, **kwargs)
    fit = fitting
    m = fit(init, x, y)
    model = m(x)
    return model, label

def plot_many_lines(x, y, yerr=None, titles=None, *models):
    if len(titles) != len(models):
        logger.error("Needs number of titles to be same as number of models!")

    fig = plt.figure(figsize=(10, 5))
    plt.errorbar(x, y, yerr=yerr, fmt='k+')
    plt.xlabel("Time")
    plt.ylabel("Flux")

    for i, model in enumerate(models):
        lab = titles[i] if titles is not None else None
        plt.plot(x, model, label=lab)

    plt.legend(loc=2)
    logger.info("done: %d models" % len(models))
    return fig

def plot_many_plots(x, y, titles=None, *models):
    n = len(models)
    size = (n*3, 4)
    fig, axes = plt.subplots(1, n+1, figsize=size, sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].plot(x, y, 'k+')
    ax[0].set_title("Original")

    for i, model in enumerate(models, 1):
        lab = titles[i-1] if titles is not None else None
        ax[i].plot(x, model, label=lab)
        # ax[i].axis('off')
        ax[i].set_title(lab)

    logger.info("done: %d models" % n)
    return fig

def determine_accuracy(x, y, model):
    var = np.square(y-model).sum()
    accuracy = var
    return accuracy

def estimate_freqs(times, max_years=13, times_per_year=0.15):
    qt_factor = np.ptp(times)/(18*90)
    year_factor = 365*qt_factor
    good_periods = np.arange(0, max_years, factor_per_year)*year_factor

    good_freqs = np.divide(1.0, good_periods)
    filtered_is = np.isposinf(good_freqs)
    good_freqs = good_freqs[np.logical_not(filtered_is)]
    return good_freqs

def run_through_models(target, x, y, yerr, do_plot=True):
    reses = [make_astropy_model(x, y, "Const1D", models.Const1D, \
                                fitting.LinearLSQFitter()) \
             , make_astropy_model(x, y, "Linear1D", models.Linear1D, \
                                  fitting.LinearLSQFitter()) \
             , make_astropy_model(x, y, "Poly1D", models.Polynomial1D, \
                                  fitting.LinearLSQFitter(), 1) \
             ]

    damped_sine_inits = [0.01, 0.001, 0.001, 0]
    lab_ds = "DampedSine " + str(damped_sine_inits)
    reses.append(make_scipy_model(x, y, lab_ds, damped_sine_model, simple_err_func, \
                                  [0.01, 0.001, 0.001, 0]))

    for freq in estimate_freqs(x, 13, 0.15):
        reses.append(make_astropy_model(x, y, "Sine1D %f" % freq, models.Sine1D, \
                                        fitting.LevMarLSQFitter(), frequency=freq))

    labs = []
    mods = []
    accs = []
    for res in reses:
        mod, lab = res
        labs.append(lab)
        mods.append(mod)
        accs.append(determine_accuracy(x, y, mod))

    if do_plot:
        fig_lines = plot_many_lines(x, y, yerr, labs, *mods)
        plt.show()
        plt.close("all")

    logger.info("done: %d models" % len(mods))
    return labs, accs

def build_errorbars(err, x, qs):
    error_list = np.zeros_like(x)
    for i in range(len(err)):
        g = np.where(qs == i)[0]
        error_list[g] += err[i]

def is_variable(targ, do_plot=True):
    target = run_photometry(targ)
    if target == 1:
        return target

    y = target.obs_flux-1
    x = target.times
    yerr = build_errorbars(target.flux_uncert, x, target.qs)

    good_indexes = np.logical_not(np.isnan(y))
    y = y[good_indexes]
    x = x[good_indexes]

    labs, accs = run_through_models(target, x, y, yerr, do_plot)
    best_i = np.argmin(accs)
    print "BEST FIT for %s is %s: %f" % (target.kic, labs[best_i], accs[best_i])

    bools = [not labs[best_i] == "Const1D"]
    result = all(bools)
    logger.info("done: %s" % result)
    return result

def main():
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

    kics = ben_kics
    kics = ["4726114", "6263983", "10087863"]

    for targ in kics:
        is_variable(targ, do_plot=True)

    return 0

if __name__ == "__main__":
    from settings import setup_main_logging, mpl_setup
    logger = setup_main_logging()
    setup_main_logging
    mpl_setup()

    main()
else:
    from settings import setup_logging
    logger = setup_logging()
