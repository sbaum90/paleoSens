import iminuit
from multiprocessing import Pool
from functools import partial
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
import importlib

from scipy.stats import chi2
import sys
import time
import copy
import os

sys.path.insert(1, '../')
from paleoSpec import CalcSpectra

# ---------------------------------------------
# Intro
# ---------------------------------------------
"""
USAGE: python3 darkdisk_reach.py [runfile]
See fiducial_darkdisk.py for format

Per default, darkdisk_reach uses a Poisson-likelihood to calculate the
sensitivity. It allows to incorporate external (Gaussian) constraints
on the nuissance parameters via the "* uncertainty*" parameters in the 
runfile.

If the optional parameter "Gaussian likelihood" is set to 
"Gaussian likelihood: Yes"
in the runfile, the sensitivity is instead calculated by assuming
a normal distribution of the number of events in each bin. Per default, 
the code will compute the variance in each bin from the Poisson error
only. However, an additional parameters 
"Relative background systematics: NUMBER" 
can be included. If this parameter is set to a value different from 0., 
the code will project the sensitivity by including an additional RELATIVE 
systematic error of the number of background events in each bin. Hence, in 
this case the variance of the number of events in the i-th bin is set by
var_i = N_i + (rel_bkg_sys * N_i^bkg)**2
where N_i is the number of events in the ith bin N_i^bkg the number of
background events in the ith bin, and rel_bkg_sys the relative systematc
error. Note that N_i and N_i^bkg, differ by the contribution of the signal 
to the Asimov data. Finally, note that if "Gaussian_likelihood" is not 
set to "Yes" the code uses a Poisson likelihood and the 
"Relative background systematics" parameter (if set) is ignored
"""

num_zero = 1e-100


def get_Spec_asimov(norm_disk,
    mass_ind,
    Spec_disk,
    Spec_nu,
    Spec_neutrons, 
    Spec_1a,
    single_alpha = True,
    neutrons = True,
):
    """
    Function to construct Asimov data
    - Inputs:
        - norm_disk: relative dark disk normalization, i.e. the WIMP
                     cross section times the disk density divided by
                     ref_SIDD * ref_SigDD
        - m: index of disk WIMP mass parameter in disk_masses
    - Outputs Spectrum of Asimov data, which has shape (N_samples, nbins)
    """
    # dark disk contribution
    Spec = norm_disk * Spec_disk[mass_ind]
    # neutrino background contribution
    Spec += np.sum(Spec_nu, axis=0)
    # neutron background contribution (if applicable)
    if neutrons:
        Spec += Spec_neutrons
    # single-alpha background contribution (if applicable)
    if single_alpha:
        Spec += Spec_1a
    # add small contribution to ensure that all entries are nonzero
    Spec += np.full_like(Spec, num_zero)
    return Spec


def get_Spec_H0(thetas,
    n_nu,
    N_samples,
    Spec_halo_interp,
    Spec_nu,
    Spec_neutrons,
    Spec_1a,
    mass_active=True,
    neutrons=True,
    single_alpha=True
):
    """
    Function to construct Spectrum under null hypothesis (halo + bkg)
    for a given choice of nuisance parameters
    - Takes array thetas as input
        - The interpretations of these parameters are similar to those
          of theta_bool (reiterated below), but with two additional
          parameters for halo normalization and WIMP mass.
        - All nuisance parameters (except the halo WIMP mass) are
          relative to the given values in Part I, i.e. norm_mass = 1
          implies the masses of the rocks are given exactly by
          sample_masses, while norm_halo = 1 implies that the halo
          WIMP cross section is exactly ref_SIDD.
    - Outputs Spectrum with shape (N_samples, nbins)
    """
    # relative neutrino flux normalization
    norm_nu = thetas[:n_nu]
    # N_samples parameters for relative rock ages
    norm_ages = thetas[n_nu : n_nu + N_samples]
    # N_samples parameters for relative sample masses (if applicable)
    if mass_active:
        norm_mass = thetas[n_nu + N_samples : n_nu + 2 * N_samples]
    # N_samples parameters for relative uranium concentrations (if applicable)
    if neutrons or single_alpha:
        norm_uranium = thetas[-N_samples - 2 : -2]
    # relative halo normalization
    norm_halo = thetas[-2]
    # halo WIMP mass
    mass = thetas[-1]
    # compute Spectrum
    Spec = norm_halo * Spec_halo_interp(mass) * norm_ages[:, None]
    Spec += np.tensordot(norm_nu, Spec_nu, axes=(0, 0)) * norm_ages[:, None]
    if neutrons:
        Spec += Spec_neutrons * norm_ages[:, None] * norm_uranium[:, None]
    if single_alpha:
        Spec += Spec_1a * norm_uranium[:, None]
    # renormalize by relative sample mass (if applicable)
    if mass_active:
        Spec *= norm_mass[:, None]
    Spec += np.full_like(Spec, num_zero)
    return Spec


# def get_Poisson_logL_H0(l10_thetas, Spec_asimov):
#     """
#     Function to compute Poisson log likelihood under null hypothesis 
#     for a given choice of nuisance parameters.
#     Note that this function expects log10 of the input parameters compared 
#     to the functions computing the Spectra above
#     This function is now obsolete, but I left it here for now in case we need it later
#     """
#     thetas = 10 ** l10_thetas
#     Spec_H0 = get_Spec_H0(thetas)
#     # Poisson contribution
#     return np.sum(Spec_asimov * np.log(Spec_H0) - Spec_H0)


def get_Gaussian_constraints_logL_H0(l10_thetas, ext_bool, ext_unc):
    """
    Function to compute log likelihood of the Gaussian constraints under 
    for a given choice of nuisance parameters.
    Note that this function expects log10 of the input parameters compared 
    to the functions computing the Spectra above
    """
    thetas = 10 ** l10_thetas
    return -0.5 * np.linalg.norm((thetas[:-2][ext_bool] - 1) / ext_unc) ** 2


def get_Poisson_logLR(l10_thetas, Spec_asimov, get_Spec_H0_partial):
    """
    Function to compute (discovery reach) Poisson log likelihood ratio  
    for a given choice of nuisance parameters.
    Note that this function expects log10 of the input parameters compared 
    to the functions computing the Spectra above
    """
    thetas = 10 ** l10_thetas
    Spec_H0 = get_Spec_H0_partial(thetas)
    return np.sum(Spec_asimov * np.log(Spec_asimov / Spec_H0) - (Spec_asimov - Spec_H0))


def get_Gaussian_logLR(
    l10_thetas, 
    Spec_asimov, 
    Spec_asimov_bkg, 
    get_Spec_H0_partial,
    rel_sys=0.01
    ):
    """
    Function to compute (discovery reach) Gaussian log likelihood ratio  
    for a given choice of nuisance parameters.
    Includes an extra relative systematic error to the variance given by  
    (rel_bkg_sys * Asimov_data_bkg)**2 per bin
    Note that this function expects log10 of the input parameters compared 
    to the functions computing the Spectra above
    """
    thetas = 10 ** l10_thetas
    Spec_H0 = get_Spec_H0_partial(thetas)
    # compute Gaussian likelihood ratio
    out = 0.5 * np.sum(
        (Spec_asimov - Spec_H0)**2
        / (Spec_asimov + (rel_sys * Spec_asimov_bkg)**2)
        )
    return out


def get_init_norm_halo(
    N_samples,
    n_nu,
    nthetas,
    Spec_halo_interp,
    Spec_nu,
    Spec_neutrons,
    Spec_1a,
    mass,
    Spec_asimov,
    mass_active=True,
    neutrons=True,
    single_alpha=True,
):
    """
    Function to compute inital guess for halo normalization,
    given a halo WIMP mass
    - The inital guess for norm_halo will be the normalization which
      gives the halo + bkg Spectrum the same number of total counts
      as the Asimov data.
    """
    # background-only Spectrum
    Spec_bkg = get_Spec_H0(np.concatenate((np.ones(nthetas), [0], [mass])),
    n_nu,
    N_samples,
    Spec_halo_interp,
    Spec_nu,
    Spec_neutrons,
    Spec_1a,
    mass_active=mass_active,
    neutrons=neutrons,
    single_alpha=single_alpha,
    )
    # difference in counts between Asimov data and background-only Spectrum
    counts_diff = np.sum(Spec_asimov - Spec_bkg)
    # number of counts in halo Spectrum
    counts_halo = np.sum(Spec_halo_interp(mass))
    if counts_halo > 0:
        return counts_diff / counts_halo
    else:
        return 1.

def get_TS(norm_disk,
    Spec_disk, 
    ref_SIDD,
    ref_SigDD,
    N_samples,
    n_nu,
    nthetas,
    ext_bool,
    ext_unc,
    nbins,
    Spec_halo_interp,
    Spec_nu,
    Spec_1a,
    Spec_neutrons,
    halo_masses,
    mass_ind, 
    Gaussian_likelihood=False,
    Relative_background_systematics=0.0,
    use_minimizer_minuit=True,
    include_bkg_nu_solar=True,
    include_bkg_nu_GSNB=True,
    include_bkg_nu_DSNB=True,
    include_bkg_nu_atm=True,
    include_bkg_rad_1a=True,
    include_bkg_rad_neutrons=True,
    mass_active=True,
):
    """
    Function to compute test statistic
    - Input parameters same as get_Spec_asimov (norm_disk and m)
    - Works by first trying a few guesses for the halo WIMP mass (and
      corresponding guesses for norm_halo), and then using the best one
      as the initial value for the optimization
    - Bounds all optimization parameters to be positive, except the halo
      WIMP mass, which must lie within the bounds of the interpolator
    - Note that technically, the test statistic is the ratio of the
      maximum likelihood under H1 to the maximum likelihood under H0.
      Since we know the maximum likelihood under H1 exactly, here we will
      compute the ratio of this maximum likelihood to the likelihood
      under H0 for fixed thetas and minimize this ratio over all thetas.
    """
    # compute asimov data set
    Spec_asimov = get_Spec_asimov(norm_disk, mass_ind, Spec_disk, Spec_nu, Spec_1a, Spec_neutrons)

    Spec_H0_partial = lambda thetas_temp: get_Spec_H0(thetas_temp,
    n_nu,
    N_samples,
    Spec_halo_interp,
    Spec_nu,
    Spec_neutrons,
    Spec_1a,
    mass_active=mass_active,
    neutrons=include_bkg_rad_neutrons,
    single_alpha=include_bkg_rad_1a,
    )
    #----------------------
    # mk function for likelihood ratio
    #----------------------
    
    if not Gaussian_likelihood:
        fun_logLR = lambda l10_thetas: 2.*(
            get_Poisson_logLR(l10_thetas, Spec_asimov, Spec_H0_partial)
            - get_Gaussian_constraints_logL_H0(l10_thetas, ext_bool, ext_unc)
            )
    else:
        Spec_asimov_bkg = get_Spec_asimov(0.0, mass_ind, Spec_disk, Spec_nu, Spec_1a, Spec_neutrons)
        fun_logLR = lambda l10_thetas: 2.*(
            get_Gaussian_logLR(l10_thetas, Spec_asimov, Spec_asimov_bkg, Spec_H0_partial, rel_sys=Relative_background_systematics)
            - get_Gaussian_constraints_logL_H0(l10_thetas, ext_bool, ext_unc)
            )
    #----------------------
    # get initial values of parameters for optimizer
    #----------------------
    logLRs = []
    l10_guess_thetas = []
    # array of halo WIMP mass guesses to try
    l10_guess_masses = np.linspace(
        np.log10(halo_masses[0]), 
        np.log10(halo_masses[-1]), 
        100
        )
    for l10_guess_mass in l10_guess_masses:
        # corresponding guess for norm_halo
        l10_guess_norm_halo = np.log10(
            get_init_norm_halo(N_samples,
            n_nu,
            nthetas,
            Spec_halo_interp,
            Spec_nu,
            Spec_neutrons,
            Spec_1a,
            10**l10_guess_mass,
            Spec_asimov,
            mass_active=mass_active,
            neutrons=include_bkg_rad_neutrons,
            single_alpha=include_bkg_rad_1a,
            ))
        # guess for all parameters (use 1s for all parameters other than 
        # last two)
        l10_guess_thetas.append(
            np.concatenate((
                np.zeros(nthetas), 
                [l10_guess_norm_halo], 
                [l10_guess_mass]
                ))
            )  
        logLRs.append(fun_logLR(l10_guess_thetas[-1]))
    # choose best mass guess
    l10_init_thetas = l10_guess_thetas[np.argmin(logLRs)]  
    #----------------------
    # set up list of parameter bounds
    #----------------------
    l10_bounds = []
    for val in l10_init_thetas[:-1]: 
        l10_bounds.append((val-10., val+20.))
    # WIMP mass bounds
    l10_bounds.append((
        np.log10(halo_masses[0]), 
        np.log10(halo_masses[-1])
        ))
    l10_bounds = np.array(l10_bounds)
    #----------------------
    # run the optimizer
    #----------------------
    if use_minimizer_minuit:
        """
        Optimize using iminuit. Strategy:
        1. Fix the parameters controlling each component of the 
           null-hypothesis model except for the sample masses and ages 
           (i.e., the normalizations of the neutrino parameters, the 
           uranium concentration, and the two parameters controlling 
           the MW-halo contribution)
        2. Order these parameters by the effect a small change (EPS)
           in the parameter has in the likelihood (DELTA_LOGLR) around 
           the inital guess (INIT_THETAS)
        3. Release the parameter causing the largest change. (or both 
           the halo mass and normalization if either of these causes 
           the largest change) and run the optimizer (MIGRAD)
        4. Release the next component, run the optimizer (it automatically 
           starts from the endpoint of the previous optimization)
        5. repeat until all parameters are free
        """
        # initialize optimizer
        optimizer = iminuit.Minuit(fun_logLR, l10_init_thetas)
        optimizer.limits = l10_bounds
        optimizer.errordef = optimizer.LIKELIHOOD
        # get indices of the positions of the background normalization 
        # parameters in theta
        theta_inds = np.concatenate((
            np.linspace(0, n_nu-1, n_nu),
            np.linspace(-N_samples-2, -2-1, N_samples*(include_bkg_rad_neutrons or include_bkg_rad_1a)),
            np.array([-666])
            ))
        # get change of test statistic with each parameter
        eps = 1e-3
        logLR_ref = np.min(logLRs)
        Delta_logLR = np.zeros(theta_inds.shape)
        for i, ind in enumerate(theta_inds):
            if ind==-666:
                l10_thetas1 = np.copy(l10_init_thetas)
                l10_thetas1[-2] += eps 
                l10_thetas2 = np.copy(l10_init_thetas)
                l10_thetas2[-1] += eps 
                Delta_logLR[i] = np.max([
                    np.abs(
                        fun_logLR(l10_thetas1)
                        - fun_logLR(l10_init_thetas)
                        ),
                    np.abs(
                        fun_logLR(l10_thetas2)
                        - fun_logLR(l10_init_thetas)
                        )
                    ])
            else:
                l10_thetas = np.copy(l10_init_thetas)
                l10_thetas[int(ind)] += eps 
                Delta_logLR[i] = np.abs(
                    fun_logLR(l10_thetas)
                    - fun_logLR(l10_init_thetas)
                    )
        # lock parameters
        for ind in theta_inds:
            if ind == -666:
                optimizer.fixed[-2] = True
                optimizer.fixed[-1] = True
            else:
                optimizer.fixed[int(ind)] = True
        # release parameters one by one and optimize
        for i in np.argsort(-Delta_logLR):
            if i == len(Delta_logLR)-1:
                optimizer.fixed[-2] = False
                optimizer.fixed[-1] = False
                optimizer.migrad()
            else:
                optimizer.fixed[int(theta_inds[i])] = False
                optimizer.migrad()
        TS = optimizer.fval
    else:  
        # default scipt optimizer
        optimizer_output = optimize.minimize(fun_logLR, 
            l10_init_thetas, 
            bounds=l10_bounds, 
            )
        TS = optimizer_output.fun
    return TS

def get_reach(Spec_disk, 
    ref_SIDD,
    ref_SigDD,
    N_samples,
    n_nu,
    nthetas,
    ext_bool,
    ext_unc,
    nbins,
    Spec_halo_interp,
    Spec_nu,
    Spec_1a,
    Spec_neutrons,
    halo_masses,
    mass_ind, 
    threshold=1e14, 
    no_reach_out_cm2Msolpc2=1e-30,
    Gaussian_likelihood=False,
    Relative_background_systematics=0.0,
    use_minimizer_minuit=True,
    include_bkg_nu_solar=True,
    include_bkg_nu_GSNB=True,
    include_bkg_nu_DSNB=True,
    include_bkg_nu_atm=True,
    include_bkg_rad_1a=True,
    include_bkg_rad_neutrons=True,
    mass_active=True,
):
    """
    Function to calculate reach for a given mass index m.
    Searches for norm_disk which returns the critical
    value of the test statistic
    Inputs:
    - m: index of WIMP mass in disk_masses
        kwargs:
        - threshold: max normalization of log-likelihood allowed
        - no_reach_out_cm2Msolpc2: value returned if the function
                                   finds no reach [cm^2.Msol.pc^-2]
    """
    # critical value of test statistic, given by Wilks' theorem 
    # (approximately 3.84)
    TS_crit = chi2.ppf(0.95, 1)
    # get smallest normalization from demanding that there
    # must be at least 1 signal event in the oldest rock
    if np.sum(Spec_disk[mass_ind][-1]) > 0:
        l10_reach_min = np.log10(1.0 / np.sum(Spec_disk[mass_ind][-1]))
    else:
        l10_reach = np.log10(no_reach_out_cm2Msolpc2 / ref_SIDD / ref_SigDD)
        return 10**l10_reach

    # get max normalization from demanding that the
    # log-likelihood of the true hypothesis is smaller than
    # THRESHOLD to avoid numerical errors
    Spec_asimov = lambda norm_disk, m: get_Spec_asimov(norm_disk, mass_ind, Spec_disk, Spec_nu, Spec_1a, Spec_neutrons)
    l10_reach_max = optimize.brentq(
        lambda l10_norm: (
            np.sum(
                Spec_asimov(10 ** l10_norm, mass_ind)
                * np.log(Spec_asimov(10 ** l10_norm, mass_ind))
                - Spec_asimov(10 ** l10_norm, mass_ind)
            )
            - threshold
        ),
        -100.0,
        100.0,
    )

    # run root_finder 10**l10_norm
    fun = lambda l10_norm: get_TS(10**l10_norm,
        Spec_disk, 
        ref_SIDD,
        ref_SigDD,
        N_samples,
        n_nu,
        nthetas,
        ext_bool,
        ext_unc,
        nbins,
        Spec_halo_interp,
        Spec_nu,
        Spec_1a,
        Spec_neutrons,
        halo_masses,
        mass_ind, 
        Gaussian_likelihood=Gaussian_likelihood,
        Relative_background_systematics=Relative_background_systematics,
        use_minimizer_minuit=use_minimizer_minuit,
        include_bkg_nu_solar=include_bkg_nu_solar,
        include_bkg_nu_GSNB=include_bkg_nu_GSNB,
        include_bkg_nu_DSNB=include_bkg_nu_DSNB,
        include_bkg_nu_atm=include_bkg_nu_atm,
        include_bkg_rad_1a=include_bkg_rad_1a,
        include_bkg_rad_neutrons=include_bkg_rad_neutrons,
        mass_active=mass_active,
        ) - TS_crit
    if fun(l10_reach_min) * fun(l10_reach_max) > 0:
        l10_reach = np.log10(no_reach_out_cm2Msolpc2 / ref_SIDD / ref_SigDD)
    else:
        l10_reach = optimize.brentq(fun, l10_reach_min, l10_reach_max)
    print(
        "reach for mass point ",
        mass_ind,
        "computed."
    )
    return 10 ** l10_reach


def get_Spec_halo(mineralname,
    Htracks,
    ref_xsec,
    readout_resolution_Aa,
    xmin_Aa,
    xmax_Aa,
    nbins,
    DM_mass,
):
    # function to calculate halo Spectrum for given WIMP mass
    calculator = CalcSpectra.CalcSpectra(mineralname, switch_keep_H=Htracks)
    return CalcSpectra.smear_and_bin(
        calculator.calc_dRdx_MW(DM_mass, ref_xsec),
        readout_resolution_Aa,
        xmin=xmin_Aa,
        xmax=xmax_Aa,
        nbins=nbins,
        logbins=True,
    )[1]

def get_Spec_disk(mineralname,
    Htracks,
    ref_xsec,
    ref_SigDD,
    vvDD,
    thetavDD,
    thetaorbitDD,
    sigvDD,
    readout_resolution_Aa,
    xmin_Aa,
    xmax_Aa,
    nbins,
    Replace_disk_spectrum_with_halo,
    params,
):
    # function to calculate disk Spectrum of each
    # crossing (indexed by i) for a given WIMP mass
    mass, i = params  # unpack disk WIMP mass and index of crossing parameters
    calculator = CalcSpectra.CalcSpectra(mineralname, switch_keep_H=Htracks)
    if not Replace_disk_spectrum_with_halo:
        return CalcSpectra.smear_and_bin(
            calculator.calc_dndx_DD(
                mass,
                ref_xsec,
                ref_SigDD,
                vvDD[i],
                thetavDD[i],
                thetaorbitDD[i],
                sigvDD,  # disk parameters
            ),
            readout_resolution_Aa,
            xmin=xmin_Aa,
            xmax=xmax_Aa,
            nbins=nbins,
            logbins=True,
        )[1]
    else:
        return CalcSpectra.smear_and_bin(
            calculator.calc_dRdx_MW(mass, ref_xsec),
            readout_resolution_Aa,
            xmin=xmin_Aa,
            xmax=xmax_Aa,
            nbins=nbins,
        )[1]

def main_runner():
    # start time (used to print time elapsed)
    t0 = time.time() 

    # Reference values to calculate DM Spectra with
    ref_SIDD = 1e-46  # reference cross-section for both halo and disk (in cm^2)
    ref_SigDD = 10.0  # reference surface density for disk (in M_sun/pc^2)
    sigvDD = 1.0  # velocity dispersion for disk (in km/s)

    # ---------------------------------------------
    # import parameters from RUNFILE
    # ---------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python3 WIMP_reach.py [runfile]")
        print("Exiting...")
        sys.exit()

    fin_params = sys.argv[1]
    run_params = importlib.import_module(fin_params)

    ages = run_params.Youngest_sample_age + np.linspace(
        0, run_params.Sample_age_spacing * (run_params.N_samples - 1), run_params.N_samples
    )

    # array of uranium concentration  [g/g]
    uranium_conc = np.full(run_params.N_samples, run_params.U_concentration)
    # masses of rocks [kg]
    sample_mass = np.full(run_params.N_samples, run_params.Sample_mass)

    # Build array theta_bool characterizing which optimization parameters are active
    #     - True indicates parameter is active, while False indicates inactive
    #     - Interpetations of each parameter listed below
    if run_params.Neutron_background or run_params.SingleAlpha_background:
        U_background = True
    else:
        U_background = False

    theta_bool = np.array(
        [run_params.Atmos_neutrino_background]  # atmospheric neutrino background normalization
        + [run_params.DSNB_background]  # DSNB background normalization
        + [run_params.GSNB_background]  # GSNB background normalization
        + [run_params.Solar_neutrino_background]  # solar neutrino background normalization
        + [True] * run_params.N_samples  # run_params.N_samples parameters for ages of rocks (always True)
        + [run_params.Optimize_mass] * run_params.N_samples  # run_params.N_samples parameters for masses of rocks
        + [U_background] * run_params.N_samples  # run_params.N_samples parameters for uranium concentrations of rocks
        # (True if either neutron or single-alpha is turned on)
    )
    # number of active parameters 
    # (not including disk normalization and disk WIMP mass)
    nthetas = np.sum(theta_bool)

    # Initialize relative uncertainties on external constraints
    #     - Initial interpretation of parameters in ext_unc same as theta_bool,
    #       but constraints on inactive parameters will be removed
    #     - If input for uncertainty is "none", uncertainty will momentarily be
    #       set to nan and then removed

    def read_unc(x):
        # shorthand function to handle "none" inputs
        if x is None:
            return np.nan
        return x

    # array of uncertainties
    ext_unc = np.array(
        [read_unc(run_params.Atmos_neutrino_flux_unc)]
        + [read_unc(run_params.DSNB_flux_unc)]
        + [read_unc(run_params.GSNB_flux_unc)]
        + [read_unc(run_params.Solar_neutrino_unc)]
        + [read_unc(run_params.SampleAge_unc)] * run_params.N_samples
        + [read_unc(run_params.SampleMass_unc)] * run_params.N_samples
        + [read_unc(run_params.U_concentration_unc)] * run_params.N_samples
    )


    # Initialize parameters for Spectrum generation
    # resolution of read-out method (in Å)
    # lower edge of first track length bin (default value for smear_and_bin used here)
    xmin = -1.0
    # upper edge of last track length bin
    xmax = 10000.0
    # halo WIMP masses for the interpolation (in GeV)
    halo_masses = np.geomspace(0.035, 1000.0, run_params.N_halo_WIMP_ms)
    # disk WIMP masses for the final limit (in GeV)
    disk_masses = np.geomspace(1.0, 1000.0, run_params.N_disk_WIMP_ms)
    # whether to use halo Spectrum in place of disk Spectrum
    if run_params.Replace_disk_spectrum_with_halo:
        # uses disk_masses (with improved resolution)
        # for interpolation if disk Spectrum is going
        # to to be replaced with the halo Spectrum
        if run_params.N_disk_WIMP_ms < 1e3:
            run_params.N_halo_WIMP_ms = run_params.N_disk_WIMP_ms ** 2
        else:
            run_params.N_halo_WIMP_ms = run_params.N_disk_WIMP_ms
        halo_masses = np.geomspace(disk_masses[0], disk_masses[-1], run_params.N_halo_WIMP_ms)
        halo_masses = np.concatenate(
            (
                np.geomspace(0.035, disk_masses[0], 100)[:-1],
                np.geomspace(disk_masses[0], disk_masses[-1], run_params.N_halo_WIMP_ms),
                np.geomspace(disk_masses[-1], 1e4, 100)[1:],
            )
        )

    # ----------
    # print message if mineral is invalid
    if run_params.Mineral not in CalcSpectra.good_mineral_list:
        print("Target mineral not recognized!")
        print("The known minerals are:")
        for good_mineral in CalcSpectra.good_mineral_list:
            print(good_mineral)
        print("Exiting...")
        exit()


    # ----------
    # print run info to commandline if Verbose
    if run_params.Verbose:
        print("Output file:", run_params.Output)
        if run_params.Cores > 1:
            print("Cores:", run_params.Cores)
        else:
            print("NOTE: Computations will not be parallelized.")
        print("Mineral:", run_params.Mineral)
        if run_params.iminuit:
            print("Optimization method: iminuit")
        else:
            print("Optimization method: Default scipy routine")
        if run_params.Gaussian_likelihood:
            print("Using Gaussian likelihood for the number of events per bin")
            if run_params.Relative_background_systematics > 0.:
                print("Including a relative systematic error ", run_params.Relative_background_systematics, " of the backgrounds")
        else:
            if run_params.Relative_background_systematics > 0.:
                print("Relative systematic error declared in RUNFILE ignored because Poisson-likelihood used")
        print("")
        print("Number of samples:", run_params.N_samples)
        print("Youngest sample age (Myr):", run_params.Youngest_sample_age)
        print("Sample age spacing (Myr):", run_params.Sample_age_spacing)
        print("Uranium concentration (per weight):", run_params.U_concentration)
        print("Sample mass (kg):", run_params.Sample_mass)
        print("")
        if not theta_bool[0]:
            print("WARNING: Atmospheric neutrino background is turned off!")
        if not theta_bool[1]:
            print("WARNING: Galactic supernova neutrino background is turned off!")
        if not theta_bool[2]:
            print("WARNING: Diffuse supernova neutrino background is turned off!")
        if not theta_bool[3]:
            print("WARNING: Solar neutrino background is turned off!")
        if not run_params.Neutron_background:
            print("WARNING: Neutron background is turned off!")
        if not run_params.SingleAlpha_background:
            print("WARNING: Single-alpha background is turned off!")
        if not run_params.Optimize_mass:
            print("NOTE: Sample mass will not be explicitly optimized over.")
        if not (np.all(theta_bool[:4]) and run_params.Optimize_mass and run_params.Neutron_background and run_params.SingleAlpha_background):
            print("")
        # print uncertainties (or lack thereof) for constraints on active parameters
        for i, index in enumerate([0, 1, 2, 3, 4, 4 + run_params.N_samples, 4 + 2 * run_params.N_samples]):
            param_name = [
                "atmospheric neutrino flux",
                "galactic supernova neutrino flux",
                "diffuse supernova neutrino flux",
                "solar neutrino flux",
                "sample age",
                "sample mass",
                "uranium concentration",
            ][i]
            if not np.isnan(ext_unc[index]) and theta_bool[index]:
                print(param_name.capitalize(), "uncertainty:", ext_unc[index])
            elif np.isnan(ext_unc[index]):
                print("NOTE: No external constraint on", param_name + ".")
        print("")
        print("Read-out resolution (Å):", run_params.Resolution)
        print("Number of track length bins:", run_params.N_track_bins)
        print("Number of halo WIMP masses (for interpolation):", run_params.N_halo_WIMP_ms)
        print("Number of disk WIMP masses (for final limit):", run_params.N_disk_WIMP_ms)
        print("Hydrogen tracks included in Spectra:", run_params.H_tracks)
        if run_params.Replace_disk_spectrum_with_halo:
            print("WARNING: Disk Spectrum is set to halo value!")
        print("")

    # ----------
    # remove constraints on inactive parameters
    ext_unc = ext_unc[theta_bool]
    # boolean array of constraints for which uncertainties were given
    ext_bool = ~np.isnan(ext_unc)
    # remove constraints for which no uncertainties were given
    ext_unc = ext_unc[ext_bool]  

    # ----------
    # load the disk crossing parameters
    crossing_params = np.loadtxt("darkdisk_crossing_params.dat")
    crossing_times = crossing_params[:, 0]  # times of disk crossings (in Myr)
    # vertical velocity of Sun relative to disk at crossings (in km/s)
    vvDD = crossing_params[:, 1]
    # angle between Sun's velocity and disk at crossings (in rad)
    thetavDD = crossing_params[:, 2] * np.pi / 180.0
    # angle between Sun's velocity and ecliptic at crossings (in rad)
    thetaorbitDD = crossing_params[:, 3] * np.pi / 180.0

    
    # --------------------------------------
    # Part II: Compute Spectra
    # --------------------------------------

    # ----------
    # Compute neutrino background Spectra
    #     - Each background Spectrum has shape (run_params.N_samples, run_params.N_track_bins)
    #     - Spectra of all active neutrino backgrounds will be stacked into 
    #       array Spec_nu

    # initialize Spectrum calculator
    calculator = CalcSpectra.CalcSpectra(run_params.Mineral, switch_keep_H=run_params.H_tracks)
    Spec_nu = []  # array of neutrino background Spectra

    # atmospheric neutrino Spectrum
    if theta_bool[0]:
        Spec_atm = (
            np.vstack(
                [
                    CalcSpectra.smear_and_bin(
                        calculator.calc_dRdx_BkgNeu_atm(),
                        run_params.Resolution,
                        xmin=xmin,
                        xmax=xmax,
                        nbins=run_params.N_track_bins,
                        logbins=True,
                    )[1]
                ]
                * run_params.N_samples
            )  # track length Spectrum per unit track length, unit time, and unit mass
            * ages[:, None]  # factor for age of rock
            * sample_mass[:, None]  # factor for mass of rock
        )
        Spec_nu.append(Spec_atm)

    # DSNB Spectrum
    if theta_bool[1]:
        Spec_DSNB = (
            np.vstack(
                [
                    CalcSpectra.smear_and_bin(
                        calculator.calc_dRdx_BkgNeu_DSNB(),
                        run_params.Resolution,
                        xmin=xmin,
                        xmax=xmax,
                        nbins=run_params.N_track_bins,
                        logbins=True,
                    )[1]
                ]
                * run_params.N_samples
            )
            * ages[:, None]
            * sample_mass[:, None]
        )
        Spec_nu.append(Spec_DSNB)

    # GSNB Spectrum
    if theta_bool[2]:
        Spec_GSNB = (
            np.vstack(
                [
                    CalcSpectra.smear_and_bin(
                        calculator.calc_dRdx_BkgNeu_GSNB(),
                        run_params.Resolution,
                        xmin=xmin,
                        xmax=xmax,
                        nbins=run_params.N_track_bins,
                        logbins=True,
                    )[1]
                ]
                * run_params.N_samples
            )
            * ages[:, None]
            * sample_mass[:, None]
        )
        Spec_nu.append(Spec_GSNB)

    # solar neutrino Spectrum
    if theta_bool[3]:
        Spec_solar = (
            np.vstack(
                [
                    CalcSpectra.smear_and_bin(
                        calculator.calc_dRdx_BkgNeu_solar(),
                        run_params.Resolution,
                        xmin=xmin,
                        xmax=xmax,
                        nbins=run_params.N_track_bins,
                        logbins=True,
                    )[1]
                ]
                * run_params.N_samples
            )
            * ages[:, None]
            * sample_mass[:, None]
        )
        Spec_nu.append(Spec_solar)

    # get number of neutrino backgrounds
    n_nu = len(Spec_nu)
    # collect neutrino background Spectra
    Spec_nu = np.array(Spec_nu)
    # ----------
    # Compute radiogenic neutron and single alpha-background Spectra

    # neutron Spectrum
    if run_params.Neutron_background:
        Spec_neutrons = (
            np.vstack(
                [
                    CalcSpectra.smear_and_bin(
                        calculator.calc_dRdx_Bkgn(1.0),
                        run_params.Resolution,
                        xmin=xmin,
                        xmax=xmax,
                        nbins=run_params.N_track_bins,
                        logbins=True,
                    )[1]
                ]
                * run_params.N_samples
            )
            * ages[:, None]
            * uranium_conc[:, None]  # factor for uranium concentration in rock
            * sample_mass[:, None]
        )
    else:
        Spec_neutrons = np.zeros((run_params.N_samples, run_params.N_track_bins))

    # single-alpha background Spectrum
    if run_params.SingleAlpha_background:
        Spec_1a = (
            np.vstack(
                [
                    calculator.smear_and_bin_1a(
                        1.0, 
                        run_params.Resolution, 
                        xmin=xmin, 
                        xmax=xmax, 
                        nbins=run_params.N_track_bins, 
                        logbins=True
                    )[1]
                ]
                * run_params.N_samples
            )
            * uranium_conc[:, None]
            * sample_mass[:, None]
        )
    else:
        Spec_1a = np.zeros((run_params.N_samples, run_params.N_track_bins))


    # parallelize Spectrum computation over all WIMP masses
    func_halo = partial(
        get_Spec_halo,
        run_params.Mineral,
        run_params.H_tracks,
        ref_SIDD,
        run_params.Resolution,
        xmin,
        xmax,
        run_params.N_track_bins,
    )
    if run_params.Cores > 1:
        with Pool(run_params.Cores) as p:
            Spec_halo_flux = p.map(func_halo, halo_masses)
            p.close()
            p.join()
    else:
        Spec_halo_flux = [func_halo(mass) for mass in halo_masses]

    # rearrange result to the right form and normalize appropriately
    Spec_halo_flux = np.array(Spec_halo_flux)
    Spec_halo_raw = Spec_halo_flux[:, None, :] * ages[:, None] * sample_mass[:, None]

    # interpolate between WIMP masses
    # note that the interpolation returns the Spectrum at the smallest (largest)
    # WIMP mass in halo_masses if one calls Spec_halo_interp(mass) for a mass
    # outside of the range of halo_masses. Since this will yield a flat
    # contribution to the likelihood if the optimizer moves outside of the mass
    # range of halo_masses, it should force it to turn back.
    Spec_halo_interp = interp1d(
        halo_masses, 
        Spec_halo_raw, 
        axis = 0,
        bounds_error = False,
        fill_value = (Spec_halo_raw[0], Spec_halo_raw[-1])
        )
    # ----------
    # Compute disk Spectra (with parallelization)

    # get maximal number of crossings
    max_crossings = np.where(crossing_times >= ages[-1])[0][0]

    # list of input parameters for Spectrum calculation
    inputs = [[mass, i] for mass in disk_masses for i in range(max_crossings)]

    # parallelize Spectrum computation over all WIMP masses
    func_disk = partial(
        get_Spec_disk,
        run_params.Mineral,
        run_params.H_tracks,
        ref_SIDD,
        ref_SigDD,
        vvDD,
        thetavDD,
        thetaorbitDD,
        sigvDD,
        run_params.Resolution,
        xmin,
        xmax,
        run_params.N_track_bins,
        run_params.Replace_disk_spectrum_with_halo,
    )
    if run_params.Cores > 1:
        with Pool(run_params.Cores) as p:
            Spec_disk_flux = p.map(func_disk, inputs)
            p.close()
            p.join()
    else:
        Spec_disk_flux = [func_disk(params) for params in inputs]

    # reformat
    Spec_disk_flux = np.array(Spec_disk_flux).reshape((run_params.N_disk_WIMP_ms, max_crossings, run_params.N_track_bins))

    # add Spectra over crossings as appropriate for each rock
    Spec_disk = np.zeros(
        (run_params.N_disk_WIMP_ms, run_params.N_samples, run_params.N_track_bins)
    )  # array of disk Spectra for each WIMP mass
    for n, age in enumerate(ages):
        # get number of relevant crossings for rock i
        crossings = np.where(crossing_times >= age)[0][0]
        # sum over relevant crossings
        Spec_disk[:, n, :] = np.sum(Spec_disk_flux[:, :crossings], axis=1) * sample_mass[n]

    print("Spectra ready:", time.time() - t0, "s")

    # ----------
    # Compute reach for each disk WIMP mass (with parallelization)
    # and write results to file
    fun_reach = partial(
        get_reach,
        Spec_disk,
        ref_SIDD,
        ref_SigDD,
        run_params.N_samples,
        n_nu,
        nthetas,
        ext_bool,
        ext_unc,
        run_params.N_track_bins,
        Spec_halo_interp,
        Spec_nu,
        Spec_1a,
        Spec_neutrons,
        halo_masses,
        include_bkg_nu_solar=run_params.Solar_neutrino_background,
        include_bkg_nu_GSNB=run_params.GSNB_background,
        include_bkg_nu_DSNB=run_params.DSNB_background,
        include_bkg_nu_atm=run_params.Atmos_neutrino_background,
        include_bkg_rad_1a=run_params.SingleAlpha_background,
        include_bkg_rad_neutrons=run_params.Neutron_background,
        Gaussian_likelihood=run_params.Gaussian_likelihood,
        Relative_background_systematics=run_params.Relative_background_systematics,
        mass_active = run_params.Optimize_mass,
        )
    if run_params.Cores > 1:
        with Pool(run_params.Cores) as p:
            reach = p.map(fun_reach, range(run_params.N_disk_WIMP_ms))
            p.close()
            p.join()
    else:
        reach = [fun_reach(mass_ind) for mass_ind in range(run_params.N_disk_WIMP_ms)]

    reach = np.array(reach)
    # reintroduce factors of reference cross-section and reference density
    reach *= ref_SIDD * ref_SigDD
    print("Optimization finished:", time.time() - t0, "s")

    output = np.vstack((disk_masses, reach)).T
    outfile = run_params.Output  # output filename
    np.savetxt(
        outfile,
        output,
        header="Mass [GeV], Cross-section * Surface Density [M_sun/pc^2*cm^2]",
    )
    print("Results written to file:", time.time() - t0, "s")
    print("")
    print("Computation complete!")

if __name__ == "__main__":
    main_runner()