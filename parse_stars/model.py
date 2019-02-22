from aperture import run_photometry
from utils import format_arr
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
from scipy import optimize

class model(object):
    """
    applies models to a light curve

    :y: y-values, light curve values
    :x: x-values, light curve times
    :yerr: y-value errors, must be same length as y
    :xerr: x-value errors, must be same length as x
    :qs: list of quarters of each light curve value
    :fmts: list of format of each point, can be length 4 or length of y-values
    """
    def __init__(self, y, x, yerr=None, xerr=None, qs=None, fmts=None):
        x = np.array(x)
        y = np.array(y)
        self.index = ~(np.isnan(y))
        self.x_dat = x[self.index]
        self.y_dat = y[self.index]
        if xerr:
            new_xerr = np.array(xerr)[self.index]
            new_xerr[np.isnan(new_xerr)] = 0
            self.xerr_dat = new_xerr
        else:
            self.xerr_dat = xerr
        if yerr:
            new_yerr = np.array(yerr)[self.index]
            new_yerr[np.isnan(new_yerr)] = 0
            self.yerr_dat = new_yerr
        else:
            self.yerr_dat = yerr

        if qs:
            self.qs_dat = np.array(qs)[self.index]
            self.fmts = self._setup_fmts(qs) \
                if (not fmts or len(fmts) != len(y)) else np.array(fmts)
            self.fmts_dat = self.fmts[self.index]
        else:
            self.qs_dat = qs
            self.fmts_dat = None

        med_y = np.array(np.median(y[:8]))
        med_i = 0
        self.med_i = med_i
        y_short = np.array(np.insert(y[8:], 0, med_y))
        x_short = np.array(np.insert(x[8:], 0, x[med_i]))
        yerr_short = np.array(np.insert(yerr[8:], 0, yerr[med_i]))
        qs_short = np.array(np.insert(qs[8:], 0, qs[med_i]))
        fmts_short = np.array(np.insert(self.fmts[8:], 0, self.fmts[med_i]))
        index_short = ~(np.isnan(y_short))
        self.x = x_short[index_short]
        self.y = y_short[index_short]
        self.qs = qs_short[index_short]
        self.fmts = fmts_short[index_short]
        self.yerr = yerr_short[index_short]
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

    def make_scipy_model(self, model_func, err_func, init_arr, bounds=(-np.inf, np.inf)):
        res = optimize.least_squares(err_func, init_arr, args=(self.x, self.y, model_func), \
                                     method="dogbox", bounds=bounds, max_nfev=2000)
        model = model_func(self.x, res.x)
        return model, res.x

    def make_astropy_model(self, model_func, fitting, var_names, *args, **kwargs):
        init = model_func(*args, **kwargs)
        fit = fitting
        m = fit(init, self.x, self.y)
        p = []
        for name in var_names:
            attr = getattr(m, name)
            p.append(attr.value)
        model = m(self.x)
        return model, p

    def plot_many_lines(self, **kwargs):
        fig = plt.figure(figsize=(10, 5))

        if self.fmts is None:
            plt.errorbar(self.x, self.y, yerr=self.yerr, fmt="k+")
        else:
            for i in range(len(self.x)):
                plt.errorbar(self.x[i], self.y[i], yerr=self.yerr[i], fmt=self.fmts[i])
        plt.xlabel("Time")
        plt.ylabel("Flux")

        for i, res in enumerate(self.reses):
            label, model, p, k = res
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
            label, model, p, k = res
            ax[i].plot(self.x, model, label=label, fmt='k+', **kwargs)
            ax[i].set_title(label)
            # ax[i].axis('off')

        logger.info("done: %d models" % n)
        return fig

    def _get_ssr(self, model, data, err):
        return np.nansum(np.square(np.divide(model-data, err)))

    def _get_bic(self, model, k):
        ssr = self._get_ssr(model, self.y, self.yerr)
        bic = ssr + k*np.log(len(model))
        return bic, np.nansum(ssr)

    def _get_jitter_ssr(self, j, model, data, yerr):
        s = np.sqrt(yerr**2 + j**2)
        ssr = self._get_ssr(model, data, s)
        d = (ssr - len(model))**2
        return d

    def _get_jitter(self, model, gs):
        init = [0.005]
        bnds = [(0, None)]
        res = optimize.minimize(self._get_jitter_ssr, init, \
                                args=(model[gs], self.y[gs], self.yerr[gs]), bounds=bnds)
        return res.x

    def _get_jitter_grid(self, model, gs):
        n = np.array(len(gs))
        is_first = True
        for j in np.arange(0, 1, 0.0001):
            s = np.sqrt(self.yerr[gs]**2 + j**2)
            ssr = self._get_ssr(model[gs], self.y[gs], s)
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
        model = self.reses[0][1]
        ran = 4 if np.any(self.qs) else 1
        for i in range(ran):
            gs = np.where(np.array(self.qs) == i)[0]
            jitter = self._get_jitter(model, gs)
            new_err = np.sqrt(self.yerr[gs]**2 + jitter**2)
            self.yerr[gs] = [err if (not (np.isnan(err) or err == 0.0)) else self.yerr[gs[i]] \
                             for i, err in enumerate(new_err)]
        self.yerr_dat = np.concatenate((np.repeat(self.yerr[0], 8), self.yerr[1:]))
        return 0

    def _determine_accuracy(self, res):
        label, model, p, k = res
        acc = self._get_bic(model, k)
        return acc

    def _sort_accuracies(self):
        bics = []
        ssrs = []
        labs = []
        self.format_res = []
        for res in self.reses:
            label, model, p, k = res
            bic, ssr = self._determine_accuracy(res)
            bics.append(bic)
            ssrs.append(ssr)
            labs.append(label)
            self.format_res += [bic, "[" + format_arr(p, ",") + "]"]
        bics = np.array(bics)
        labs = np.array(labs)

        idxs = np.argsort(bics)
        sorted_labs = labs[idxs]
        sorted_bics = bics[idxs]
        sorted_ssrs = ssrs[idxs]

        unique_labs = []
        unique_idxs = []
        unique_ssrs = []
        for i, lab in enumerate(sorted_labs):
            if lab not in unique_labs:
                unique_labs.append(lab)
                unique_idxs.append(idxs[i])
        return labs[unique_idxs], bics[unique_idxs]

    def _get_best_accuracies(self):
        bics = []
        ssrs = []
        self.format_res = []
        for res in self.reses[:3]:
            label, model, p, k = res
            bic, ssr = self._determine_accuracy(res)
            bics.append(bic)
            ssrs.append(ssr)
            self.format_res += [bic, "[" + format_arr(p, ",") + "]"]

        best_bic = float("inf")
        cur_bic = cur_ssr = cur_p = None
        for res in self.reses[3:]:
            label, model, p, k = res
            bic, ssr = self._determine_accuracy(res)
            if bic <= best_bic:
                cur_bic = bic
                cur_ssr = ssr
                cur_p = p
        bics.append(cur_bic)
        ssrs.append(cur_ssr)
        self.format_res += [bic, "[" + format_arr(cur_p, ",") + "]"]


        self.bics = np.array(bics)
        self.ssrs = np.array(ssrs)

        best_i = np.argmin(self.bics)
        best_bic = self.bics[best_i]
        best_ssr = self.ssrs[best_i]
        best_lab = self.reses[best_i][0]
        return best_i, best_lab, best_bic, best_ssr

    def _estimate_freqs(self, times, min_years=2, max_years=13, times_per_year=0.25):
        qt_factor = 1
        year_factor = 365*qt_factor
        good_periods = np.arange(min_years, max_years, times_per_year)*year_factor

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
        model, p = model_res
        self.reses.append((label, model, p, k_params))
        return 0

    def run_const_model(self):
        return self._setup_res("Const1D", \
                               self.make_astropy_model(models.Const1D, \
                                                       fitting.LinearLSQFitter(), \
                                                       ["amplitude"]), 1)

    def run_linear_model(self):
        return self._setup_res("Linear1D", \
                               self.make_astropy_model(models.Linear1D, \
                                                       fitting.LinearLSQFitter(), \
                                                       ["slope", "intercept"]), 2)

    def run_parabola_model(self):
        return self._setup_res("Parabola1D", \
                               self.make_astropy_model(models.Polynomial1D, \
                                                       fitting.LinearLSQFitter(), \
                                                       ["c0", "c1", "c2"], 2), 3)

    def run_sines_model(self):
        for freq in self._estimate_freqs(self.x, 2, 13, 0.25):
            for amp in self._estimate_amps(self.y, 5):
                sine_label = "Sine1D"
                self._setup_res(sine_label, \
                                self.make_scipy_model(self._sine_model, self._simple_err_func, \
                                                      [amp, freq, 0], \
                                                      ([-np.inf, -np.inf, -np.inf], \
                                                       [np.inf, 0.002, np.inf])), 3)
        return 0

    def run_through_models(self):
        self.reses = []
        self.run_const_model()
        self.run_linear_model()
        self.run_parabola_model()
        self.run_sines_model()
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
                model = self.make_scipy_model(self._sine_model, self._simple_err_func, \
                                              [amp, freq, 0], (None, np.inf))
                k = 3
        return self._get_ssr(model), self._get_bic(mode1l, k)

    def is_variable(self):
        if np.all(np.isnan(self.y)):
            self.format_res = str([np.nan])
            return False, "Unsure", np.nan, np.nan

        self.run_const_model()
        self.fix_errors()
        self.run_through_models()

        best_i, best_lab, best_bic, best_ssr = self._get_best_accuracies()
        bools = [not best_lab == "Const1D"]
        result = all(bools)
        logger.info("done: %s, %s" % (result, best_lab))
        return result, best_lab, best_bic, best_ssr

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
