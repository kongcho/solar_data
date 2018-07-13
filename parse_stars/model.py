import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import numpy as np

from aperture import run_photometry
from settings import setup_main_logging
logger = setup_main_logging()

def build_errorbars(err, x, qs):
    error_list = np.zeros_like(x)
    for i in range(len(err)):
        g = np.where(qs == i)[0]
        error_list[g] += err[i]

def make_1D_model(x, y, model, fitting, label="", *args, **kwargs):
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
    reses = []
    reses.append(make_1D_model(x, y, models.Sine1D, fitting.LevMarLSQFitter(), "Sine1D", frequency=0.001))
    reses.append(make_1D_model(x, y, models.Polynomial1D, fitting.LevMarLSQFitter(), "Polynomial1D", 3))
    reses.append(make_1D_model(x, y, models.Linear1D, fitting.LinearLSQFitter(), "Linear1D"))
    reses.append(make_1D_model(x, y, models.Const1D, fitting.LinearLSQFitter(), "Const1D"))

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

    print labs
    print accs

    return accs

def main():
    targ = "3236788"
    target = run_photometry(targ)

    y = target.obs_flux-1
    x = target.times
    yerr = build_errorbars(target.flux_uncert, x, target.qs)
    run_through_models(target, x, y, yerr)

    return 0

if __name__ == "__main__":
    main()
