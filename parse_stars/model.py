import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import numpy as np

from aperture import run_photometry
from settings import setup_main_logging
logger = setup_main_logging()

from scipy import optimize

def sine_model(x, inits):
    return inits[0]*np.sin(2*np.pi*inits[1]*x + inits[2])

def log_model(x, inits):
    return inits[0] + inits[1]*np.log(x)

def simple_err_func(x, y, fit_func, init_arr):
    return fit_func(init_arr, x) - y

def make_scipy_model(x, y, fit_func, err_func, init_arr):
    p, success = optimize.leastsq(err_func, init_arr[:], args=(x, y))
    model = fit_func(p, x)
    return model

def make_astropy_model(x, y, model, fitting, label="", *args, **kwargs):
    init = model(*args, **kwargs)
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

def run_through_models(target, x, y, yerr):
    reses = [make_astropy_model(x, y, models.Const1D, fitting.LinearLSQFitter(), "Const1D") \
             # , make_astropy_model(x, y, models.Linear1D, fitting.LinearLSQFitter(), "Linear1D") \
             # , make_astropy_model(x, y, models.Polynomial1D, fitting.LevMarLSQFitter(), "Polynomial1D", 3)) \
             , make_astropy_model(x, y, models.PowerLaw1D, fitting. LevMarLSQFitter(), "PowerLaw1D", alpha=0.01) \
             ]

    # for freq in np.arange(0, 0.01, 0.001):
    #     reses.append(make_astropy_model(x, y, models.Sine1D, fitting.LevMarLSQFitter(), "Sine1D %f" % freq, frequency=freq))

    mods = []
    labs = []
    accs = []
    for res in reses:
        model, label = res
        mods.append(model)
        labs.append(label)
        accs.append(determine_accuracy(x, y, model))

    fig_lines = plot_many_lines(x, y, yerr, labs, *mods)

    plt.show()
    plt.close("all")

    best_i = np.argmin(accs)

    print "BEST FIT for %s is %s: %f" % (target.kic, labs[best_i], accs[best_i])

    return accs

def build_errorbars(err, x, qs):
    error_list = np.zeros_like(x)
    for i in range(len(err)):
        g = np.where(qs == i)[0]
        error_list[g] += err[i]

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

    for targ in kics:
        target = run_photometry(targ)
        if target == 1:
            continue

        y = target.obs_flux-1
        x = target.times
        yerr = build_errorbars(target.flux_uncert, x, target.qs)
        run_through_models(target, x, y, yerr)

    return 0

if __name__ == "__main__":
    main()
