import sys
import importlib
from importlib import reload
import os
import argparse
import math
import glob
import astropy
from astropy.io import fits
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt
import time
import scipy as sp
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import emcee
import corner
from scipy.optimize import curve_fit
import multiprocessing
import gc
from tqdm import *
import matplotlib
import matplotlib.pyplot as plt
from mpi4py import MPI
from mpipool import MPIPool
from emcee.autocorr import AutocorrError, function_1d

# from multiprocessing import Pool
from schwimmbad import MPIPool

# pool = MPIPool()

# suppressing warnings
import warnings

warnings.filterwarnings("ignore")  # setting ignore as a parameter
import logging

logging.getLogger().setLevel(logging.ERROR)

# plt.style.use('ggplot')
plt.style.use("default")
plt.rcParams["font.family"] = "Arial"

config_cleaned_lc_directory = "/Users/thomasmoore/Desktop/SNIPER_DEV/ATLAS_TDEs/"
MJD_minus = 500
MJD_plus = 500
nwalkers = 30
nsteps = 5000
progress = True
plot = True

final_run_walker_multiplier = 2
final_run_step_multiplier = 4


parser = argparse.ArgumentParser()

parser.add_argument(
    "-f",
    "--file",
    help="txt file containing a list of IAU names which SNIPER.py will work on",
    dest="file",
    type=argparse.FileType("r"),
)

parser.add_argument(
    "-o",
    "--output",
    help="output directory (default cwd)",
    default=os.getcwd(),
    dest="output_dir",
)

args = parser.parse_args()

output_dir = args.output_dir
output_dir = output_dir + "/SNIPER_Output"

# Creating output directories if they do no already exist

if os.path.isdir(output_dir + "/corner") == False:
    print("No corner plot directory - making one!")
    os.makedirs(output_dir + "/corner/")

if os.path.isdir(output_dir + "/overplot/") == False:
    print("No overplot plot directory - making one!")
    os.makedirs(output_dir + "/overplot/")


if os.path.isdir(output_dir + "/scatter/") == False:
    print("No scatter plot directory - making one!")
    os.makedirs(output_dir + "/scatter/")


IAU_list = pd.read_csv(args.file, header=None)
IAU_list.columns = ["IAU_NAME", "t_guess"]

print(IAU_list.t_guess)


global x_global, y_global, y_err_global


def def_global(x, y, y_err):
    global x_global
    global y_global
    global y_err_global
    x_global = np.array(x).astype("float64")
    y_global = np.array(y).astype("float64")
    y_err_global = np.array(y_err).astype("float64")


def Lambobstorest(lambda_obs, Z):
    lambda_rest = np.divide(lambda_obs, (1.0 + Z))
    return lambda_rest


def Lambresttoobs(lambda_rest, Z):
    lambda_obs = np.multiply(lambda_rest, (1.0 + Z))
    return lambda_obs


def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier


def round_down(n, decimals=0):
    multiplier = 10**decimals
    return math.floor(n * multiplier) / multiplier


def janks_to_mag(jank):
    mag = -2.5 * np.log10((jank * 1e-6) / 3631)
    return mag


def magtoflux(mag, band):
    if band == "R":
        wavelength = 6580.0

    if band == "r":
        wavelength = 6170.0

    if band == "g":
        wavelength = 4754.0

    if band == "B":
        wavelength = 4450.0

    if band == "i":
        wavelength = 8000.0

    if band == "z":
        wavelength = 9665.0

    if band == "u":
        wavelength = 3580.0

    if band == "ATLAS_o":
        wavelength = 6866.26

    if band == "ATLAS_c":
        wavelength = 5408.66

    if band == "PS1_W":
        wavelength = 6579

    print(wavelength)
    flux = ((2.99 * 10**18) / (wavelength**2)) * 10 ** (-((mag + 48.60) / 2.5))
    return (wavelength, flux)


def tardis_sim(config, luminosity, time, z):
    tardis_config = config
    tardis_config.supernova.luminosity_requested = luminosity
    tardis_config.supernova.time_explosion = time
    sim = run_tardis(
        tardis_config,
        show_convergence_plots=False,
        show_progress_bars=True,
        virtual_packet_logging=True,
    )
    spectrum = sim.runner.spectrum
    spectrum_virtual = sim.runner.spectrum_virtual
    tardisflux_virtual = spectrum.luminosity_to_flux(
        spectrum_virtual.luminosity_density_lambda, cosmo.luminosity_distance(z)
    )
    # spectrum_integrated = sim.runner.spectrum_integrated
    # tardisflux_integrated = spectrum.luminosity_to_flux(spectrum_integrated.luminosity_density_lambda,cosmo.luminosity_distance(z))
    wavelength = spectrum.wavelength
    flux = tardisflux_virtual
    return (wavelength, flux, sim)


def NTT_OBS(input_spec):
    bin_width = 7
    new_disp_grid = (
        np.arange(
            np.min(input_spec.spectral_axis.value),
            np.max(input_spec.spectral_axis.value),
            bin_width,
        )
        * u.AA
    )
    fluxcon = FluxConservingResampler(extrapolation_treatment="zero_fill")
    binned_spec = fluxcon(input_spec, new_disp_grid)

    NTT_B = SpectralElement.from_file("NTT_filters/NTT_B.dat")
    NTT_V = SpectralElement.from_file("NTT_filters/NTT_V.dat")
    NTT_R = SpectralElement.from_file("NTT_filters/NTT_R.dat")
    NTT_i = SpectralElement.from_file("NTT_filters/NTT_Gunn_i.dat")

    obs_B = Observation(binned_spec, NTT_B, force="extrap")
    flux_B = synphot.units.convert_flux(obs_B.binset, obs_B.binflux, "flam")

    obs_V = Observation(binned_spec, NTT_V, force="extrap")
    flux_V = synphot.units.convert_flux(obs_V.binset, obs_V.binflux, "flam")

    obs_R = Observation(binned_spec, NTT_R, force="extrap")
    flux_R = synphot.units.convert_flux(obs_R.binset, obs_R.binflux, "flam")

    obs_i = Observation(binned_spec, NTT_i, force="extrap")
    flux_i = synphot.units.convert_flux(obs_i.binset, obs_i.binflux, "flam")

    # mag_B = obs_B.effstim()
    f, ax = plt.subplots(dpi=300)
    ax.plot(binned_spec.wavelength, binned_spec.flux)
    ax.plot(obs_B.binset, flux_B, color="b", label="B")
    ax.plot(obs_V.binset, flux_V, color="violet", label="V")
    ax.plot(obs_R.binset, flux_R, color="r", label="R")
    ax.plot(obs_i.binset, flux_i, color="grey", label="i")
    ax.legend()
    ax.set_xlim(
        np.min(input_spec.spectral_axis.value) * 0.9,
        np.max(input_spec.spectral_axis.value) * 1.1,
    )

    print("\n NTT magnitudes \n")

    print(f" Effstim B filter =  {obs_B.effstim(u.ABmag):.3f}")
    print(f" Effstim V filter = {obs_V.effstim(u.ABmag):.3f} ")
    print(f" Effstim R filter = {obs_R.effstim(u.ABmag):.3f} ")
    print(f" Effstim i filter = {obs_i.effstim(u.ABmag):.3f} ")

    return 0


cwd = os.getcwd()


def ATLAS_OBS(input_spec):
    binned_spec = input_spec

    for line in input_spec.wavelength.value:
        if math.isnan(line) == True:
            print("Found a nan in wavelength - this will cause an error")

    for line in input_spec.flux.value:
        if math.isnan(line) == True:
            print("Found a nan in flux - this will cause an error")

    ATLAS_c = SpectralElement.from_file("Misc_Atlas.cyan.dat")
    ATLAS_o = SpectralElement.from_file("Misc_Atlas.orange.dat")

    f, ax = plt.subplots(dpi=300)
    ax.plot(binned_spec.wavelength, binned_spec.flux, label="spectrum", color="k")

    try:
        obs_c = Observation(binned_spec, ATLAS_c, force="extrap")
        flux_c = synphot.units.convert_flux(obs_c.binset, obs_c.binflux, "flam")
        print(f"Synthetic ATLAS c filter \t=\t {obs_c.effstim(u.ABmag):.3f} ")
        ax.plot(obs_c.binset, flux_c, color="cyan", label="c")
    except Exception:
        print("No ATLAS c overlap!")
        pass

    try:
        obs_o = Observation(binned_spec, ATLAS_o, force="extrap")
        flux_o = synphot.units.convert_flux(obs_o.binset, obs_o.binflux, "flam")
        ax.plot(obs_o.binset, flux_o, color="orange", label="o")
        print(f"Synthetic ATLAS o filter \t=\t {obs_o.effstim(u.ABmag):.3f} ")
    except Exception:
        print("No ATLAS o overlap!")
        pass

    ax.set_xlabel("Observed Wavelength " + r"$\AA$")
    ax.set_ylabel("Flux")
    ax.legend()
    ax.set_xlim(
        np.min(input_spec.spectral_axis.value) * 0.9,
        np.max(input_spec.spectral_axis.value) * 1.1,
    )
    return 0


def PST_OBS(input_spec):
    cwd = os.getcwd()

    # binned_spec = input_spec
    for line in input_spec.wavelength.value:
        if math.isnan(line) == True:
            print("Found a nan in wavelength - this will cause an error")

    for line in input_spec.flux.value:
        if math.isnan(line) == True:
            print("Found a nan in flux - this will cause an error")

    binned_spec = input_spec

    PST_g = SpectralElement.from_file(cwd + "/PST_filters/PAN-STARRS_PS1.z.dat")
    PST_r = SpectralElement.from_file(cwd + "/PST_filters/PAN-STARRS_PS1.r.dat")
    PST_w = SpectralElement.from_file(cwd + "/PST_filters/PAN-STARRS_PS1.w.dat")
    PST_open = SpectralElement.from_file(cwd + "/PST_filters/PAN-STARRS_PS1.open.dat")
    PST_i = SpectralElement.from_file(cwd + "/PST_filters/PAN-STARRS_PS1.i.dat")
    PST_z = SpectralElement.from_file(cwd + "/PST_filters/PAN-STARRS_PS1.z.dat")
    PST_y = SpectralElement.from_file(cwd + "/PST_filters/PAN-STARRS_PS1.y.dat")

    f, ax = plt.subplots(dpi=300)
    ax.plot(binned_spec.wavelength, binned_spec.flux, label="Spectrum")

    print("\n Pan-STARRS magnitudes \n")
    # g-band
    obs_g = Observation(binned_spec, PST_g, force="extrap")

    try:
        obs_g = Observation(binned_spec, PST_g, force="extrap")
        flux_g = synphot.units.convert_flux(obs_g.binset, obs_g.binflux, "flam")
        print(flux_g)
        print(f"Synthetic PST g filter \t=\t  {obs_g.effstim(u.ABmag):.3f}")
        ax.plot(obs_g.binset, flux_g, color="b", label="g")
    except Exception:
        pass

    # r-band
    try:
        obs_r = Observation(binned_spec, PST_r, force="extrap")
        flux_r = synphot.units.convert_flux(obs_r.binset, obs_r.binflux, "flam")
        print(f"Synthetic PST r filter \t=\t  {obs_r.effstim(u.ABmag):.3f} ")
        ax.plot(obs_r.binset, flux_r, color="r", label="r")
    except Exception:
        pass

    # w-band
    try:
        obs_w = Observation(binned_spec, PST_w, force="extrap")
        flux_w = synphot.units.convert_flux(obs_w.binset, obs_w.binflux, "flam")
        print(f"Synthetic PST w filter \t=\t  {obs_w.effstim(u.ABmag):.3f} ")
        ax.plot(obs_w.binset, flux_w, color="grey", label="w")
    except Exception:
        pass

    # Open
    try:
        obs_open = Observation(binned_spec, PST_open, force="extrap")
        flux_open = synphot.units.convert_flux(
            obs_open.binset, obs_open.binflux, "flam"
        )
        print(f"Synthetic PST open \t=\t {obs_open.effstim(u.ABmag):.3f} ")
        ax.plot(obs_open.binset, flux_open, color="yellow", label="open")
    except Exception:
        pass

    # i-band
    try:
        obs_i = Observation(binned_spec, PST_i, force="extrap")
        flux_i = synphot.units.convert_flux(obs_i.binset, obs_i.binflux, "flam")
        print(f"Synthetic PST i filter \t=\t  {obs_i.effstim(u.ABmag):.3f} ")
        ax.plot(obs_i.binset, flux_i, color="violet", label="i")
    except Exception:
        pass

    # z-band
    try:
        obs_z = Observation(binned_spec, PST_z, force="extrap")
        flux_z = synphot.units.convert_flux(obs_z.binset, obs_z.binflux, "flam")
        print(f"Synthetic PST z filter \t=\t {obs_z.effstim(u.ABmag):.3f} ")
        ax.plot(obs_z.binset, flux_z, color="green", label="z")
    except Exception:
        pass

    # y-band
    try:
        obs_y = Observation(binned_spec, PST_y, force="extrap")
        flux_y = synphot.units.convert_flux(obs_y.binset, obs_y.binflux, "flam")
        print(f"Synthetic PST y filter \t=\t {obs_y.effstim(u.ABmag):.3f} ")
        ax.plot(obs_y.binset, flux_y, color="brown", label="y")
    except Exception:
        print("No PST y overlap")
        pass

    ax.set_xlabel("Observed Wavelength [Angstrom]")
    ax.set_ylabel("Flux")
    ax.legend()
    ax.set_xlim(
        np.min(input_spec.spectral_axis.value) * 0.9,
        np.max(input_spec.spectral_axis.value) * 1.1,
    )

    return 0


def bazin(t, A, B, T_rise, T_fall, t0):
    quoitent = A * np.exp(-(t - t0) / T_fall)
    divisor = 1 + np.exp(-(t - t0) / T_rise)
    constant = B
    return (quoitent / divisor) + constant


def baz_tmax(t0, T_rise, T_fall):
    return t0 + T_rise * np.log((T_fall / T_rise) - 1)


def rise_mjd_fit(t, a, T_exp_pow, n):
    y = np.where(t <= T_exp_pow, 0, a * (t - T_exp_pow) ** n)
    return y


def chisq_bazin(p):
    A, B, T_rise, T_fall, t0 = p[0], p[1], p[2], p[3], p[4]
    return np.sum(
        ((y_global - (bazin(x_global, A, B, T_rise, T_fall, t0))) / y_err_global) ** 2
    )


def lnpriorline_bazin(p):
    A, B, T_rise, T_fall, t0 = p
    if (
        0 < A < 1e6
        and -100 < B < +1000
        and 1 < T_rise < 3000
        and 1 < T_fall < 300
        and 0 < t0
        and T_rise / T_fall < 1
    ):
        return 0.0
    return -np.inf


def lnlikeline_bazin(p):
    chisq = chisq_bazin(p)
    return -0.5 * chisq


def lnprobline_bazin(p):
    lp = lnpriorline_bazin(p)
    if math.isnan(lp):
        return -np.inf
    if not np.isfinite(lp):
        return -np.inf
    ln_like = lnlikeline_bazin(p)
    if math.isnan(ln_like):
        return -np.inf
    return lp + ln_like


# Doing the same treatment for the fireball model
#  Inputs  = t, a, T_exp_pow, n
def chisq_fireball(p):
    a, T_exp_pow, n = p[0], p[1], p[2]
    return np.sum(
        ((y_global - (rise_mjd_fit(x_global, a, T_exp_pow, n))) / y_err_global) ** 2
    )


def lnpriorline_fireball(p, t_min, t_max):
    a, T_exp_pow, n = p
    if 1 < a < 2 * np.max(y_global) and t_min < T_exp_pow < t_max and 1.8 < n < 2.2:
        return 0
    return -np.inf


def lnlikeline_fireball(p):
    chisq = chisq_fireball(p)
    return -0.5 * chisq


def lnprobline_fireball(p, t_min, t_max):
    lp = lnpriorline_fireball(p, t_min, t_max)
    if not np.isfinite(lp):
        return -np.inf
    ln_like = +lnlikeline_fireball(p)
    if math.isnan(ln_like):
        return -np.inf
    return lp + ln_like


def bazin(x, A, B, T_rise, T_fall, t0):
    quoitent = A * np.exp(-(x - t0) / T_fall)
    divisor = 1 + np.exp(-(x - t0) / T_rise)
    constant = B
    return (quoitent / divisor) + constant


def baz_tmax(t0, T_rise, T_fall):
    return t0 + T_rise * np.log((T_fall / T_rise) - 1)


def baz_tmax(t0, T_rise, T_fall):
    return t0 + T_rise * np.log((T_fall / (T_rise) - 1))


def rise_mjd_fit(t, a, T_exp_pow, n):
    y = np.where(t <= T_exp_pow, 0, a * (t - T_exp_pow) ** n)
    return y


def AB_mag_err(flux, dflux):
    mag_err = 2.5 / np.log(10.0) * abs(dflux) / abs(flux)
    return mag_err


def flux_to_ABmag(flux):
    mag = 2.5 * (23 - np.log10(float(flux) * 1e-6)) - 48.6
    return mag


def fit_bazin(**kwargs):
    priors = kwargs.get("priors", [0.5 * np.max(y_global), 0, 5, 10, np.mean(x_global)])
    # priors = np.array(priors)
    nwalkers = kwargs.get("nwalkers", int(100))
    nsteps = kwargs.get("nsteps", int(500))
    progress = kwargs.get("progress", True)
    plot = kwargs.get("plot", False)
    object = kwargs.get("object", "")

    ndim = 5
    nwalkers, ndim = int(nwalkers), int(5)
    nsteps = int(int(nsteps))
    # pos = priors + 1 * np.random.randn(nwalkers, ndim)
    # A, B, T_rise, T_fall, t0

    pos = np.zeros((nwalkers, ndim))
    pos1 = float(priors[0]) + 10 * np.random.randn(nwalkers)
    pos2 = float(priors[1]) + 1 * np.random.randn(nwalkers)
    pos3 = float(priors[2]) + 5 * np.random.randn(nwalkers)
    pos4 = float(priors[3]) + 5 * np.random.randn(nwalkers)
    pos5 = float(priors[4]) + 50 * np.random.randn(nwalkers)
    pos = [pos1, pos2, pos3, pos4, pos5]
    pos = np.transpose(pos)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobline_bazin)
    sampler.run_mcmc(pos, nsteps, progress=progress)

    # with MPIPool() as pool:
    #    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobline_bazin, pool=pool)
    #    sampler.run_mcmc(pos, nsteps, progress=True)

    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(
        discard=int(nsteps * 0.4), flat=True, thin=int(nsteps * 0.01)
    )

    best_params = np.zeros(ndim)
    for i in range(ndim):
        best_params[i] = np.percentile(flat_samples[:, i], [50])
    A, B, T_rise, T_fall, t0 = best_params
    bazin_maximum = baz_tmax(t0, T_rise, T_fall)

    if plot == True:
        labels = ["A", "B", "T_rise", "T_fall", "t0"]

        plt.figure(dpi=300)
        fig = corner.corner(
            flat_samples,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
        )
        plt.savefig(output_dir + "/corner/" + object + "_bazin_corner_plot.png")

        plt.figure(dpi=200)
        fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = labels
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.01)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i - 1])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.savefig(output_dir + "/scatter/" + object + "_bazin_scatter.png")
        plt.close()

        # overplotting on graph
        plt.figure(dpi=300)
        inds = np.random.randint(len(flat_samples), size=100)
        x0 = np.linspace(np.min(x), np.max(x), 300)
        for ind in inds:
            sample = flat_samples[ind]
            plt.plot(x0, bazin(x0, *sample), "deepskyblue", alpha=0.1)
        plt.plot([], [], color="deepskyblue", alpha=0.9, label="Bazin Evaluations")

        plt.errorbar(
            x_global,
            y_global,
            y_err_global,
            fmt=".",
            color="black",
            capsize=0,
            label="ATLAS o-band",
        )

        plt.ylim(min(y_global) * 0.9, np.max(y_global) * 1.1)
        A = flat_samples[:, 0]
        B = flat_samples[:, 1]
        T_rise = flat_samples[:, 2]
        T_fall = flat_samples[:, 3]
        t0 = flat_samples[:, 4]

        t_max_samples = baz_tmax(t0, T_rise, T_fall)
        flux_max_samples = bazin(t_max_samples, A, B, T_rise, T_fall, t0)
        mcmc_flux = np.mean(np.array(flux_max_samples))
        mcmc_flux_err = np.std(np.array(flux_max_samples))

        time_max_mcmc = np.nanmean(np.array(t_max_samples))
        time_max_mcmc_err = np.nanstd(np.array(t_max_samples))

        best_params = np.zeros(ndim)
        lower_quartile = np.zeros(ndim)
        upper_quartile = np.zeros(ndim)

        for i in range(ndim):
            best_params[i] = np.percentile(flat_samples[:, i], [50])
            lower_quartile[i] = np.percentile(flat_samples[:, i], [16])
            upper_quartile[i] = np.percentile(flat_samples[:, i], [84])

        A, B, T_rise, T_fall, t0 = best_params
        A_upper, B_upper, T_rise_upper, T_fall_upper, t0_upper = upper_quartile
        A_lower, B_lower, T_rise_lower, T_fall_lower, t0_lower = lower_quartile
        print(f"t max = {time_max_mcmc}")
        plt.ylabel(r" Flux Density [$\rm \mu Jy$]")
        plt.xlabel(r"time [mjd]")
        plt.legend(frameon=False)
        plt.vlines(
            bazin_maximum,
            -50,
            np.max(y_global) * 1.3,
            color="gray",
            alpha=0.9,
            linestyle="--",
        )
        plt.savefig(output_dir + "/overplot/" + object + "_bazin_overplot.png")

        plt.close()

        print("best params", best_params)
        gc.collect()
        return (
            time_max_mcmc,
            time_max_mcmc_err,
            best_params,
            upper_quartile,
            lower_quartile,
            t_max_samples,
            flux_max_samples,
            flat_samples,
        )


def fit_fireball(**kwargs):
    priors = kwargs.get("priors", [1, np.mean(x_global), 1])
    # print(f'No priors given = assumning {priors}')
    nwalkers = kwargs.get("nwalkers", int(1000))
    nsteps = kwargs.get("nsteps", int(5000))
    progress = kwargs.get("progress", True)
    plot = kwargs.get("plot", False)
    object = kwargs.get("object", "")

    t_min = np.min(x_global)
    t_max = np.max(x_global)

    # dispersion = np.max( ((abs(np.mean(x) - t_min)), (abs(np.mean(x) - t_max))))
    dispersion = 50
    ndim = 3
    nwalkers, ndim = int(nwalkers), int(ndim)
    nsteps = int(int(nsteps))

    #  a, T_exp_pow, n
    pos = np.zeros((nwalkers, ndim))
    pos[:, 0] = float(priors[0]) + 30 * np.random.randn(nwalkers)
    pos[:, 1] = float(priors[1]) + 300 * np.random.randn(nwalkers)
    pos[:, 2] = float(priors[2]) + 5 * np.random.randn(nwalkers)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprobline_fireball, args=(t_min, t_max)
    )
    sampler.run_mcmc(pos, nsteps, progress=progress)
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(
        discard=int(nsteps * 0.4), thin=int(nsteps * 0.01), flat=True
    )

    labels = ["A", "b", "c"]
    inds = np.random.randint(len(flat_samples), size=100)

    # Doing a second MCMC for the fireball
    mcmc = np.percentile(flat_samples[:, 1], [16, 50, 84])
    time_explode = mcmc[1]

    if plot == True:
        labels = ["a", "T_explode", "n"]

        plt.figure(dpi=300)
        fig = corner.corner(
            flat_samples,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
        )
        plt.savefig(output_dir + "/corner/" + object + "_fireball_corner.png")

        plt.figure(dpi=200)
        fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = labels
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.01)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i - 1])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.savefig(output_dir + "/scatter/" + object + "_bazin_scatter.png")
        plt.close()

        # overplotting on graph
        x0 = np.linspace(np.min(x_global), np.max(x_global), 300)
        plt.figure(dpi=300)
        for ind in inds:
            sample = flat_samples[ind]
            plt.plot(x0, rise_mjd_fit(x0, *sample), "mediumorchid", alpha=0.1)
        plt.errorbar(
            lightcurve_data["MJD"],
            lightcurve_data["uJy"],
            yerr=lightcurve_data["duJy"],
            fmt=".k",
            capsize=0,
            label="rising points",
        )
        plt.ylim(-10, np.max(y_global))
        plt.errorbar(x_global, y_global, yerr, fmt=".", color="grey", capsize=0)
        plt.plot([], [], color="mediumorchid", alpha=0.9, label="Fireball Evaluations")
        plt.vlines(time_explode, 0, 100, color="k")
        plt.ylabel(r" Flux Density [$\rm \mu Jy$]")
        plt.xlabel(r"time [mjd]")
        plt.legend(frameon=False)
        plt.savefig(output_dir + "/overplot/" + object + "_fireball_overplot.png")

        plt.close()

    best_params = np.zeros(ndim)
    lower_quartile = np.zeros(ndim)
    upper_quartile = np.zeros(ndim)
    for i in range(ndim):
        best_params[i] = np.percentile(flat_samples[:, i], [50])
        lower_quartile[i] = np.percentile(flat_samples[:, i], [16])
        upper_quartile[i] = np.percentile(flat_samples[:, i], [84])

    a, T_exp_pow, n = best_params
    a_upper, T_exp_pow_upper, n_upper = upper_quartile
    a_lower, T_exp_pow_lower, n_lower = lower_quartile
    gc.collect()
    return best_params, upper_quartile, lower_quartile, flat_samples


SNIPER_OUTPUT = pd.DataFrame()

print("starting loop")

for object in tqdm(IAU_list["IAU_NAME"], leave=False):
    t_guess = IAU_list.loc[IAU_list["IAU_NAME"] == object, "t_guess"]
    print(f"Working on {object}")
    print(f"t_guess = {float(t_guess)}")
    f = []
    f = config_cleaned_lc_directory + object + "/" + object + ".o.1.00days.lc.txt"
    # print(f)
    cols = [
        [
            "MJD",
            "m",
            "dm",
            "uJy",
            "duJy",
            "F",
            "err",
            "chi/N",
            "RA",
            "Dec",
            "x",
            "y",
            "maj",
            "min",
            "phi",
            "apfit",
            "mag5sig",
            "Sky",
            "Obs",
            "mask",
        ]
    ]
    df = []
    df = pd.read_csv(f, delim_whitespace=True)
    df = df.filter(("MJD", "uJy", "duJy"), axis=1)
    plt.plot(
        df.MJD,
        df.uJy,
    )
    df.drop(df[df.duJy > 50].index, inplace=True)
    # df = df.dropna()
    x = np.array(df.MJD)
    if math.isnan(t_guess) is True:
        max_y = np.argmax(savgol_filter(df.uJy, 10, 3))
        savgol_first_guess = x[max_y]
        t_guess = savgol_first_guess

    df_cut_min = float(t_guess) - MJD_minus
    df_cut_max = float(t_guess) + MJD_plus
    df_new = df.dropna(how="any", axis=0)

    lightcurve_data = df_new.loc[
        (df["MJD"].astype("float64") >= df_cut_min)
        & (df["MJD"].astype("float64") <= df_cut_max)
    ]
    x, y, yerr = (
        lightcurve_data["MJD"].astype("float64"),
        lightcurve_data["uJy"].astype("float64"),
        lightcurve_data["duJy"].astype("float64"),
    )

    def_global(x, y, yerr)

    bazin_results = fit_bazin(
        progress=progress,
        plot=plot,
        object=object,
        nwalkers=nwalkers,
        nsteps=nsteps,
    )
    bazin_results = fit_bazin(
        priors=bazin_results[2],
        progress=progress,
        plot=plot,
        object=object,
        nwalkers=nwalkers * final_run_walker_multiplier,
        nsteps=nsteps * final_run_step_multiplier,
    )

    A, B, T_rise, T_fall, t0 = bazin_results[2]

    print("hello")
    print(baz_tmax(t0, T_rise, T_fall))

    # bazin_results = fit_bazin(x,y,yerr, priors = bazin_results[2], progress = True, plot = True, object = '22qh', nwalkers= 1000, nsteps = 1000)
    # removing lightucurve after max light

    lightcurve_data = df_new.loc[
        (df["MJD"].astype("float64") <= baz_tmax(t0, T_rise, T_fall))
    ]

    lightcurve_data = df_new.loc[
        (df["MJD"].astype("float64") >= baz_tmax(t0, T_rise, T_fall) - 200)
    ]
    print(f"t_max = {bazin_results[0]}")

    # lightcurve_data = lightcurve_data.drop(
    #    lightcurve_data[
    #        lightcurve_data["MJD"] <= (np.nanmean(bazin_results[0]) - 200)
    #    ].index
    # )

    x, y, yerr = (
        lightcurve_data["MJD"].astype(float),
        lightcurve_data["uJy"].astype(float),
        lightcurve_data["duJy"].astype(float),
    )

    plt.figure()
    plt.plot(x, y)
    plt.savefig(object + ".png")
    def_global(x, y, yerr)
    print(np.nanmean(bazin_results[0]))
    first_guess = np.nanmean(bazin_results[0])
    fireball_results = fit_fireball(
        priors=[np.max(y_global), first_guess, 2],
        progress=progress,
        plot=plot,
        object=object,
        nwalkers=nwalkers * 3,
        nsteps=nsteps * 3,
    )

    fireball_results = fit_fireball(
        priors=fireball_results[0],
        progress=progress,
        plot=plot,
        object=object,
        nwalkers=nwalkers * 3 * final_run_walker_multiplier,
        nsteps=nsteps * 3 * final_run_step_multiplier,
    )

    bazin_max = bazin_results[0]
    bazin_max_err = bazin_results[1]
    t_explode = fireball_results[0][1]
    t_explode_lower = fireball_results[0][1] - fireball_results[1][1]
    t_explode_upper = fireball_results[0][1] - fireball_results[2][1]
    risetime = bazin_max - t_explode
    risetime_err_lower = t_explode_lower - bazin_max_err
    risetime_err_upper = t_explode_upper + bazin_max_err

    mean_flux = np.nanmean(bazin_results[4])
    std_flux = np.nanstd(bazin_results[4])

    # time_max_mcmc,time_max_mcmc_err,best_params,upper_quartile,lower_quartile,t_max_samples,flux_max_samples,t_samples,
    T_rise = bazin_results[2][2]
    T_rise_lower = bazin_results[2][2] - bazin_results[3][2]
    T_rise_upper = bazin_results[2][2] - bazin_results[4][2]
    T_fall = bazin_results[2][3]
    T_fall_lower = bazin_results[2][3] - bazin_results[3][3]
    T_fall_upper = bazin_results[2][3] - bazin_results[4][3]
    T_explode = fireball_results[0][1]
    T_explode_lower = fireball_results[0][1] - fireball_results[1][1]
    T_explode_upper = fireball_results[0][1] - fireball_results[2][1]
    T_max = bazin_max
    T_max_err = bazin_max_err

    results_dict = {
        "TNS Name": object,
        "risetime": risetime,
        "risetime_upper": risetime_err_upper,
        "risetime_lower": risetime_err_lower,
        "T_rise": T_rise,
        "T_rise_upper": T_rise_upper,
        "T_rise_lower": T_rise_lower,
        "T_fall": T_fall,
        "T_fall_upper": T_fall_upper,
        "T_fall_lower": T_fall_lower,
        "T_explode": T_explode,
        "T_explode_lower": T_explode_lower,
        "T_explode_upper": T_explode_upper,
        "T_max": T_max,
        "T_max_err": T_max_err,
    }
    SNIPER_OUTPUT = SNIPER_OUTPUT.append(results_dict, ignore_index=True)

    print("T_rise", T_rise)
    print("T_rise_lower", T_rise_lower)
    print("T_rise_upper", T_rise_upper)
    print("T_fall", T_fall)
    print("T_fall_lower", T_fall_lower)
    print("T_fall_upper", T_fall_upper)
    print("T_explode", T_explode)
    print("T_explode_lower", T_explode_lower)
    print("T_explode_upper", T_explode_upper)
    print("T_max", T_max)
    print("T_max_err", T_max_err)

    from IPython.display import display, Math

    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(risetime, abs(risetime_err_lower), risetime_err_upper, str(object))
    display(Math(txt))

    SNIPER_OUTPUT.to_csv(output_dir + "/SNIPER_OUTPUT.txt", index=False)
    plt.close("all")
    gc.collect()
