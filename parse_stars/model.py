"""
FUNCTIONS THAT DETERMINES A STAR'S VARIABILITY FROM LIGHT CURVE
"""

from aperture import run_photometry

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
from scipy import optimize

class model(object):
    def __init__(self, y, x, yerr=None, xerr=None, qs=None, fmts=None):
        x = np.array(x)
        y = np.array(y)
        self.index = ~(np.isnan(y))
        self.x = x[self.index]
        self.y = y[self.index]
        self.xerr = np.array(xerr)[self.index] if xerr else xerr
        self.yerr = np.array(yerr)[self.index] if yerr else yerr
        qs = None
        if qs:
            self.qs = np.array(qs)[self.index]
            self.fmts = self._setup_fmts(qs)[self.index] if not fmts else np.array(fmts)
        else:
            self.qs = qs
            self.fmts = None
        self.reses = []

    def _setup_fmts(self, qs):
        fmt = ['ko', 'rD', 'b^', 'gs']
        fmts = []
        for q in qs:
            for i in range(4):
                if q == i:
                    fmts.append(fmt[i])
        return np.array(fmts)

    def _damped_sine_model(self, x, inits):
        return inits[0]*np.exp(-1*inits[1]*x)*np.sin(2*np.pi*inits[2]*x + inits[3])

    def _sine_model(self, x, inits):
        return inits[0]*np.sin(2*np.pi*inits[1]*x + inits[2])

    def _log_model(self, inits):
        return inits[0] + inits[1]*np.log(self.x)

    def _simple_err_func(self, inits, x, y, model_func):
        return model_func(x, inits) - y

    def make_scipy_model(self, model_func, err_func, init_arr):
        p, success = optimize.leastsq(err_func, init_arr, args=(self.x, self.y, model_func), \
                                      maxfev=2000)
        model = model_func(self.x, p)
        return model

    def make_astropy_model(self, model_func, fitting, *args, **kwargs):
        init = model_func(*args, **kwargs)
        fit = fitting
        m = fit(init, self.x, self.y)
        model = m(self.x)
        return model

    def plot_many_lines(self, **kwargs):
        fig = plt.figure(figsize=(10, 5))

        if np.any(self.fmts):
            for i in range(len(self.x)):
                plt.errorbar(self.x[i], self.y[i], yerr=self.yerr[i], fmt=self.fmts[i])
        else:
            plt.errorbar(self.x, self.y, yerr=self.yerr, fmt="k+")
        plt.xlabel("Time")
        plt.ylabel("Flux")

        for i, res in enumerate(self.reses):
            label, model, k = res
            plt.plot(self.x, model, label=label, linewidth=0.5, **kwargs)

        plt.legend(loc="upper right")
        logger.info("done: %d models" % len(self.reses))
        return fig

    def plot_many_plots(self, **kwargs):
        n = len(self.reses)
        size = (n*3, 4)
        fig, axes = plt.subplots(1, n+1, figsize=size, sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].plot(self.x, self.y, 'k+', **kwargs)
        ax[0].set_title("Original")

        for i, res in enumerate(self.reses, 1):
            label, model, k = res
            ax[i].plot(self.x, model, label=label, fmt='k+', **kwargs)
            ax[i].set_title(label)
            # ax[i].axis('off')

        logger.info("done: %d models" % n)
        return fig

    def _get_ssr(self, model, data, err):
        return np.square(np.divide(model-data, err))

    def _get_bic(self, model, k):
        ssr = self._get_ssr(model, self.y, self.yerr)
        return np.nansum(ssr) + k*np.log(len(model))

    def _determine_accuracy(self, res):
        label, model, k = res
        return self._get_bic(model, k)

    def _get_jitter_grid(self, model, gs):
        n = np.array(len(gs))
        is_first = True
        for j in np.arange(0, 0.01, 0.0001):
            s = np.sqrt(self.yerr[gs]**2 + j**2)
            ssr = np.sum(self._get_ssr(model[gs], self.y[gs], s))
            d = np.abs(ssr - n)
            if is_first:
                best_j = j
                best_d = d
                is_first = False
            if d < best_d:
                best_j = j
                best_d = d
        return best_j

    def fix_errors(self):
        ssrs = [np.nansum(self._get_ssr(res[1], self.y, self.yerr)) for res in self.reses]
        ssrs = [np.nan if ssr == 0.0 else ssr for ssr in ssrs]
        best_i = np.nanargmin(ssrs) if not np.all(np.isnan(ssrs)) else np.argmin(ssrs)
        model = self.reses[best_i][1]
        ran = 4 if np.any(self.qs) else 1
        for i in range(ran):
            gs = np.where(np.array(self.qs) == i)[0]
            jitter = self._get_jitter_grid(model, gs)
            new_err = np.sqrt(self.yerr[gs]**2 + jitter**2)
            self.yerr[gs] = [err if not np.isnan(err) else self.yerr[gs[i]] \
                             for i, err in enumerate(new_err)]
        return 0

    def _get_best_accuracies(self):
        accs = [self._determine_accuracy(res) for res in self.reses]
        best_i = np.argmin(accs)
        best_lab = self.reses[best_i][0]
        return best_i, best_lab

    def _estimate_freqs(self, times, max_years=13, times_per_year=0.15):
        qt_factor = np.ptp(times)/(18*90)
        year_factor = 365*qt_factor
        good_periods = np.arange(1, max_years, times_per_year)*year_factor

        good_freqs = np.divide(1.0, good_periods)
        filtered_is = np.isposinf(good_freqs)
        good_freqs = good_freqs[np.logical_not(filtered_is)]
        return good_freqs

    def _estimate_amps(self, ys, n):
        min_val = 0.0005
        max_ys = np.nanmax(np.abs(ys))
        range_ys = max_ys - min_val
        return np.arange(min_val, max_ys, range_ys/n)

    def _setup_res(self, label, model_res, k_params):
        self.reses.append((label, model_res, k_params))
        return 0

    def run_through_models(self):
        self._setup_res("Const1D", \
                        self.make_astropy_model(models.Const1D, fitting.LinearLSQFitter()), 1)
        self._setup_res("Linear1D", \
                        self.make_astropy_model(models.Linear1D, fitting.LinearLSQFitter()), 2)
        self._setup_res("Parabola1D", \
                        self.make_astropy_model(models.Polynomial1D, fitting.LinearLSQFitter(), 2),\
                        3)

        # damped_sine_inits = [0.01, 0.001, 0.001, 0]
        # self._setup_res("DampedSine", \
        #                 self.make_scipy_model(self._damped_sine_model, self._simple_err_func, \
        #                                       damped_sine_inits), len(damped_sine_inits))

        # for freq in self._estimate_freqs(self.x, 13, 0.125):
        #     self._setup_res("Sine1D %f" % freq, \
        #                     self.make_astropy_model(models.Sine1D, fitting.LevMarLSQFitter(), \
        #                                             frequency=freq), 3)

        for freq in self._estimate_freqs(self.x, 13, 0.25):
            for amp in self._estimate_amps(self.y, 5):
                self._setup_res("Sine1D %f %f" % (freq, amp), \
                                self.make_scipy_model(self._sine_model, self._simple_err_func, \
                                                      [amp, freq, 0]), 3)
        logger.info("done")
        return 0

    def run_model(self, label):
        model = []
        k = 1
        if label == "Const1D":
            model = self.make_astropy_model(models.Const1D, fitting.LinearLSQFitter())
            k = 1
        if label == "Linear1D":
            model = self.make_astropy_model(models.Linear1D, fitting.LinearLSQFitter())
            k = 2
        if label == "Parabola1D":
            model = self.make_astropy_model(models.Polynomial1D, fitting.LinearLSQFitter(), 2)
            k = 3
        else:
            labels = label.split(" ")
            if labels[0] == "Sine1D":
                freq = float(labels[1])
                amp = float(labels[2])
                model = self.make_scipy_model(self._sine_model, self._simple_err_func, [amp, freq, 0])
                k = 3
        return self._get_ssr(model), self._get_bic(model, k)

    def is_variable(self):
        self.run_through_models()

        self.fix_errors()
        best_i, best_lab = self._get_best_accuracies()

        bools = [not best_lab == "Const1D"]
        result = all(bools)
        logger.info("done: %s, %s" % (result, best_lab))
        return result, best_lab

def main():
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
