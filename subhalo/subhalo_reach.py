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

sys.path.insert(1, '../')
from paleoSpec import CalcSpectra

# ---------------------------------------------
# Intro
# ---------------------------------------------
"""
USAGE: subhalo_reach.py [runfile]
See subhalo_default_runfile.txt for format

Per default, subhalo_reach uses a Poisson-likelihood to calculate the
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


def get_spec_asimov(
    norm_sh,
    mass,
    bsh,
    spec_sh_interp,
    spec_halo_interp,
    spec_1a,
    spec_neutrons,
    spec_nu,
    indep_index,
    Neutron_background=True,
    SingleAlpha_background=True,
):
    """
    Function to construct Asimov data
    - Inputs:
        - norm_sh: relative subhalo normalization, i.e. the WIMP
                   cross section divided by ref_SIDD
        - mass, Msh, csh, bsh, vsh : as in get_spec_sh
    - Outputs spectrum of Asimov data, which has shape (run_params.N_samples, run_params.N_track_bins)
    """
    # subhalo contribution
    spec = norm_sh * spec_sh_interp(bsh)[indep_index]
    # MW contribution
    spec += norm_sh * spec_halo_interp(mass)  # I don't really understand this shape
    # neutrino background contribution
    spec += np.sum(spec_nu, axis=0)
    # neutron background contribution (if applicable)
    if Neutron_background:
        spec += spec_neutrons
    # single-alpha background contribution (if applicable)
    if SingleAlpha_background:
        spec += spec_1a
    # add small contribution to ensure that all entries are nonzero
    spec += np.full_like(spec, num_zero)
    return spec


def get_spec_H0(
    thetas,
    n_nu,
    N_samples,
    spec_halo_interp,
    spec_nu,
    spec_neutrons,
    spec_1a,
    Optimize_mass=True,
    SingleAlpha_background=True,
    Neutron_background=True,
):
    """
    Function to construct spectrum under null hypothesis (halo + bkg)
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
    - Outputs spectrum with shape (run_params.N_samples, run_params.N_track_bins)
    """
    # relative neutrino flux normalization
    norm_nu = thetas[:n_nu]
    # nrocks parameters for relative rock ages
    norm_ages = thetas[n_nu : n_nu + N_samples]
    # nrocks parameters for relative sample masses (if applicable)
    if Optimize_mass:
        norm_mass = thetas[n_nu + N_samples : n_nu + 2 * N_samples]
    # nrocks parameters for relative uranium concentrations (if applicable)
    if Neutron_background or SingleAlpha_background:
        norm_uranium = thetas[-N_samples - 2 : -2]
    # relative halo normalization
    norm_halo = thetas[-2]
    # WIMP mass
    mass = thetas[-1]
    # compute spectrum
    spec = norm_halo * spec_halo_interp(mass) * norm_ages[:, None]
    spec += np.tensordot(norm_nu, spec_nu, axes=(0, 0)) * norm_ages[:, None]
    if Neutron_background:
        spec += spec_neutrons * norm_ages[:, None] * norm_uranium[:, None]
    if SingleAlpha_background:
        spec += spec_1a * norm_uranium[:, None]
    # renormalize by relative sample mass (if applicable)
    if Optimize_mass:
        spec *= norm_mass[:, None]
    spec += np.full_like(spec, num_zero)
    return spec


def get_Poisson_logL_H0(l10_thetas, spec_asimov):
    """
    Function to compute Poisson log likelihood under null hypothesis
    for a given choice of nuisance parameters.
    Note that this function expects log10 of the input parameters compared
    to the functions computing the spectra above
    This function is now obsolete, but I left it here for now in case we need it later
    """
    thetas = 10 ** l10_thetas
    spec_H0 = get_spec_H0(thetas)
    # Poisson contribution
    return np.sum(spec_asimov * np.log(spec_H0) - spec_H0)


def get_Gaussian_constraints_logL_H0(l10_thetas, ext_bool, ext_unc):
    """
    Function to compute log likelihood of the Gaussian constraints under
    for a given choice of nuisance parameters.
    Note that this function expects log10 of the input parameters compared
    to the functions computing the spectra above
    """
    thetas = 10 ** l10_thetas
    return -0.5 * np.linalg.norm((thetas[:-2][ext_bool] - 1) / ext_unc) ** 2


def get_Poisson_logLR(l10_thetas, spec_asimov, spec_H0_partial):
    """
    Function to compute (discovery reach) Poisson log likelihood ratio
    for a given choice of nuisance parameters.
    Note that this function expects log10 of the input parameters compared
    to the functions computing the spectra above
    """
    thetas = 10 ** l10_thetas
    spec_H0 = spec_H0_partial(thetas)
    return np.sum(spec_asimov * np.log(spec_asimov / spec_H0) - (spec_asimov - spec_H0))


def get_Gaussian_logLR(
    l10_thetas, spec_asimov, spec_asimov_bkg, spec_H0_partial, rel_sys=0.0
):
    """
    Function to compute (discovery reach) Gaussian log likelihood ratio
    for a given choice of nuisance parameters.
    Includes an extra relative systematic error to the variance given by
    (rel_bkg_sys * Asimov_data_bkg)**2 per bin
    Note that this function expects log10 of the input parameters compared
    to the functions computing the spectra above
    """
    thetas = 10 ** l10_thetas
    spec_H0 = spec_H0_partial(thetas)
    # compute Gaussian likelihood ratio
    out = 0.5 * np.sum(
        (spec_asimov - spec_H0) ** 2 / (spec_asimov + (rel_sys * spec_asimov_bkg) ** 2)
    )
    return out


def get_init_norm_halo(
    mass,
    n_nu,
    nthetas,
    N_samples,
    spec_halo_interp,
    spec_nu,
    spec_neutrons,
    spec_1a,
    spec_asimov,
    Optimize_mass=True,
    SingleAlpha_background=True,
    Neutron_background=True,
):
    """
    Function to compute inital guess for halo normalization,
    given a halo WIMP mass
    - The inital guess for norm_halo will be the normalization which
      gives the halo + bkg spectrum the same number of total counts
      as the Asimov data.
    """
    # background-only spectrum
    spec_bkg = get_spec_H0(
        np.concatenate((np.ones(nthetas), [0], [mass])),
        n_nu,
        N_samples,
        spec_halo_interp,
        spec_nu,
        spec_neutrons,
        spec_1a,
        Optimize_mass=Optimize_mass,
        SingleAlpha_background=SingleAlpha_background,
        Neutron_background=Neutron_background,
    )
    # difference in counts between Asimov data and background-only spectrum
    counts_diff = np.sum(spec_asimov - spec_bkg)
    # number of counts in halo spectrum
    counts_halo = np.sum(spec_halo_interp(mass))
    if counts_halo > 0:
        return counts_diff / counts_halo
    else:
        return 1.0


def get_TS(
    norm_sh,
    mass,
    # Msh,
    # csh,
    bsh,
    vsh,
    spec_sh_interp,
    # bsh_range,
    N_samples,
    n_nu,
    nthetas,
    ext_bool,
    ext_unc,
    # N_track_bins,
    spec_halo_interp,
    spec_1a,
    spec_neutrons,
    spec_nu,
    indep_index,
    wimp_masses,
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
    - Input parameters same as get_spec_asimov
    - Works by first trying a few guesses for the WIMP mass (and
      corresponding guesses for norm_halo), and then using the
      best one as the initial value for the optimization
    - Bounds all optimization parameters to be positive, except the
      WIMP mass, which must lie within the bounds of the interpolator
    - Note that technically, the test statistic is the ratio of the
      maximum likelihood under H1 to the maximum likelihood under H0.
      Since we know the maximum likelihood under H1 exactly, here we will
      compute the ratio of this maximum likelihood to the likelihood
      under H0 for fixed thetas, and minimize this ratio over all thetas.
    """
    # compute asimov data set
    spec_asimov_func = lambda norm_sh, WIMP_m, bsh_temp: get_spec_asimov(
        norm_sh,
        WIMP_m,
        bsh_temp,
        spec_sh_interp,
        spec_halo_interp,
        spec_1a,
        spec_neutrons,
        spec_nu,
        indep_index,
        Neutron_background=include_bkg_rad_neutrons,
        SingleAlpha_background=include_bkg_rad_1a,
    )
    spec_asimov = spec_asimov_func(norm_sh, mass, bsh)

    spec_H0_partial = lambda thetas_temp: get_spec_H0(
        thetas_temp,
        n_nu,
        N_samples,
        spec_halo_interp,
        spec_nu,
        spec_neutrons,
        spec_1a,
        Optimize_mass=mass_active,
        SingleAlpha_background=include_bkg_rad_1a,
        Neutron_background=include_bkg_rad_neutrons,
    )
    # ----------------------
    # mk function for likelihood ratio
    # ----------------------
    if not Gaussian_likelihood:
        fun_logLR = lambda l10_thetas: 2.0 * (
            get_Poisson_logLR(l10_thetas, spec_asimov, spec_H0_partial)
            - get_Gaussian_constraints_logL_H0(l10_thetas, ext_bool, ext_unc)
        )
    else:
        spec_asimov_bkg = spec_asimov_func(0.0, mass, vsh)
        fun_logLR = lambda l10_thetas: 2.0 * (
            get_Gaussian_logLR(
                l10_thetas,
                spec_asimov,
                spec_asimov_bkg,
                spec_H0_partial,
                rel_sys=Relative_background_systematics,
            )
            - get_Gaussian_constraints_logL_H0(l10_thetas, ext_bool, ext_unc)
        )
    # ----------------------
    # get initial values of parameters for optimizer
    # ----------------------
    logLRs = []
    l10_guess_thetas = []
    # array of WIMP mass guesses to try
    l10_guess_masses = np.linspace(
        np.log10(wimp_masses[0]), np.log10(wimp_masses[-1]), 100
    )
    for l10_guess_mass in l10_guess_masses:
        # corresponding guess for norm_halo
        l10_guess_norm_halo = np.log10(
            get_init_norm_halo(
                10 ** l10_guess_mass,
                n_nu,
                nthetas,
                N_samples,
                spec_halo_interp,
                spec_nu,
                spec_neutrons,
                spec_1a,
                spec_asimov,
                Optimize_mass=mass_active,
                SingleAlpha_background=include_bkg_rad_1a,
                Neutron_background=include_bkg_rad_neutrons,
            )
        )
        # guess for all parameters (use 1s for all parameters other than
        # last two)
        l10_guess_thetas.append(
            np.concatenate((np.zeros(nthetas), [l10_guess_norm_halo], [l10_guess_mass]))
        )
        logLRs.append(fun_logLR(l10_guess_thetas[-1]))
    # choose best mass guess
    l10_init_thetas = l10_guess_thetas[np.argmin(logLRs)]
    # ----------------------
    # set up list of parameter bounds
    # ----------------------
    l10_bounds = []
    for val in l10_init_thetas[:-1]:
        l10_bounds.append((val - 10.0, val + 20.0))
    # WIMP mass bounds
    l10_bounds.append((np.log10(wimp_masses[0]), np.log10(wimp_masses[-1])))
    l10_bounds = np.array(l10_bounds)
    # ----------------------
    # run the optimizer
    # ----------------------
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
        5. Repeat until all parameters are free
        """
        # initialize optimizer
        optimizer = iminuit.Minuit(fun_logLR, l10_init_thetas)
        optimizer.limits = l10_bounds
        optimizer.errordef = optimizer.LIKELIHOOD
        # get indices of the positions of the background normalization
        # parameters in theta
        theta_inds = np.append(np.arange(nthetas), np.array([-666]))  # np.concatenate((
        # np.linspace(0, n_nu-1, n_nu),
        # np.linspace(-run_params.N_samples-2, -2-1, run_params.N_samples*(run_params.Neutron_background or run_params.SingleAlpha_background)),
        # np.array([-666])
        # ))
        # get change of test statistic with each parameter
        eps = 1e-3
        logLR_ref = np.min(logLRs)
        Delta_logLR = np.zeros(theta_inds.shape)
        for i, ind in enumerate(theta_inds):
            if ind == -666:
                l10_thetas1 = np.copy(l10_init_thetas)
                l10_thetas1[-2] += eps
                l10_thetas2 = np.copy(l10_init_thetas)
                l10_thetas2[-1] += eps
                Delta_logLR[i] = np.max(
                    [
                        np.abs(fun_logLR(l10_thetas1) - fun_logLR(l10_init_thetas)),
                        np.abs(fun_logLR(l10_thetas2) - fun_logLR(l10_init_thetas)),
                    ]
                )
            else:
                l10_thetas = np.copy(l10_init_thetas)
                l10_thetas[int(ind)] += eps
                Delta_logLR[i] = np.abs(
                    fun_logLR(l10_thetas) - fun_logLR(l10_init_thetas)
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
            if i == len(Delta_logLR) - 1:
                optimizer.fixed[-2] = False
                optimizer.fixed[-1] = False
                optimizer.migrad()
            else:
                optimizer.fixed[int(theta_inds[i])] = False
                optimizer.migrad()
        TS = optimizer.fval
    else:
        # default scipy optimizer
        optimizer_output = optimize.minimize(
            fun_logLR,
            l10_init_thetas,
            bounds=l10_bounds,
        )
        TS = optimizer_output.fun
    return TS


def get_reach_bsh(
    spec_sh_interp,
    norm_sh,
    bsh_range,
    N_samples,
    n_nu,
    nthetas,
    ext_bool,
    ext_unc,
    WIMP_m,
    spec_halo_interp,
    spec_1a,
    spec_neutrons,
    spec_nu,
    indep_range,
    wimp_masses,
    vsh,
    threshold=1e14,
    bsh_cutoff=1e-4,
    no_reach_out_bsh=1e-4,
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
    Function to calculate reach for impact parameter
    Searches for bsh which returns the critical
    value of the test statistic
    Inputs same as get_TS (except bsh)
    kwargs:
        - threshold: max normalization of log-likelihood allowed
        - bsh_cutoff: lower cutoff for impact parameter to avoid
                      numerical errors in CalcSpectra
        - no_reach_out_bsh: value returned if the function
                            finds no reach [km/s]
    """
    # independent variable is vsh with index 5
    # dependent variable is impact parameter with index 4
    # adjust bsh_cutoff if it is below the interpolation range
    bsh_lower = bsh_range[0]
    bsh_upper = bsh_range[-1]
    bsh_cutoff = min(bsh_cutoff, bsh_lower)
    # critical value of test statistic, given by Wilks' theorem
    # (approximately 3.84)
    TS_crit = chi2.ppf(0.95, 1)
    # get max impact parameter from demanding that there
    # must be at least 1 signal event in the oldest rock
    indep_index = np.where(indep_range == vsh)[0][0]
    # print(norm_sh)
    # quit()
    sh_sum_m1 = (
        lambda l10_bsh: np.sum(norm_sh * spec_sh_interp(10 ** l10_bsh)[indep_index, -1])
        - 1.0
    )
    if sh_sum_m1(np.log10(bsh_cutoff)) * sh_sum_m1(np.log10(bsh_upper)) > 0:
        l10_reach_max = np.log10(bsh_upper)
    else:
        l10_reach_max = optimize.brentq(
            sh_sum_m1,
            np.log10(bsh_cutoff),
            np.log10(bsh_upper),
        )
    # get smallest impact parameter from demanding that the
    # log-likelihood of the true hypothesis is smaller than
    # THRESHOLD to avoid numerical errors (or bsh_cutoff if
    # it never exceeds THRESHOLD)
    spec_asimov_func = lambda bsh: get_spec_asimov(
        norm_sh,
        WIMP_m,
        bsh,
        spec_sh_interp,
        spec_halo_interp,
        spec_1a,
        spec_neutrons,
        spec_nu,
        indep_index,
        Neutron_background=include_bkg_rad_neutrons,
        SingleAlpha_background=include_bkg_rad_1a,
    )
    asimov_LL = lambda bsh_temp: np.sum(
        spec_asimov_func(bsh_temp) * np.log(spec_asimov_func(bsh_temp))
        - spec_asimov_func(bsh_temp)
    )
    if asimov_LL(bsh_cutoff) <= threshold:
        l10_reach_min = np.log10(bsh_cutoff)
    else:
        l10_reach_min = optimize.brentq(
            lambda l10_bsh: asimov_LL(10 ** l10_bsh) - threshold,
            np.log10(bsh_cutoff),
            np.log10(bsh_upper),
        )
    # run root_finder
    fun = (
        lambda l10_bsh: get_TS(
            norm_sh,
            WIMP_m,
            10 ** l10_bsh,
            vsh,
            spec_sh_interp,
            N_samples,
            n_nu,
            nthetas,
            ext_bool,
            ext_unc,
            spec_halo_interp,
            spec_1a,
            spec_neutrons,
            spec_nu,
            indep_index,
            wimp_masses,
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
        )
        - TS_crit
    )
    if fun(l10_reach_min) < 0:
        l10_reach = np.log10(no_reach_out_bsh)
    elif fun(l10_reach_max) > 0:
        l10_reach = np.log10(bsh_upper)
    else:
        l10_reach = optimize.brentq(fun, l10_reach_min, l10_reach_max)
    # print index in indep_range of these parameter choices
    print("reach for params", indep_index, "computed.")
    sys.stdout.flush()
    return 10 ** l10_reach


def get_Spec_halo(
    mineralname,
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


def get_spec_sh(
    mineralname,
    Htracks,
    ref_xsec,
    readout_resolution_Aa,
    xmin_Aa,
    xmax_Aa,
    nbins,
    T_closest_approach,
    ages,
    delta_ages,
    params,
    Replace_subhalo_spectrum_with_halo=False,
):
    # function to calculate subhalo spectrum given
    # - mass: WIMP mass (in GeV)
    # - Msh: subhalo mass (in M_sun)
    # - csh: concentration parameter
    # - bsh: impact parameter (in pc)
    # - vsh: relative velocity (in km/s)
    mass, Msh, csh, bsh, vsh = params
    spec_sh = np.zeros((len(ages), nbins))
    calculator = CalcSpectra.CalcSpectra(mineralname, switch_keep_H=Htracks)
    if not Replace_subhalo_spectrum_with_halo:
        # spectrum corresponding to a crossing (integration time is taken to be delta_ages)
        spec_crossing = CalcSpectra.smear_and_bin(
            calculator.calc_dndx_sh(
                mass, ref_xsec, Msh, csh, bsh, vsh, -delta_ages / 2.0, delta_ages / 2.0
            ),
            readout_resolution_Aa,
            xmin=xmin_Aa,
            xmax=xmax_Aa,
            nbins=nbins,
            logbins=True,
        )[1]
    else:
        # use MW spectrum if run_params.Replace_subhalo_spectrum_with_halo is True
        spec_crossing = CalcSpectra.smear_and_bin(
            calculator.calc_dRdx_MW(mass, ref_xsec),
            run_params.readout_resolution_Aa,
            xmin=xmin_Aa,
            xmax=xmax_Aa,
            nbins=nbins.N_track_bins,
            logbins=True,
        )[1]
    for n, age in enumerate(ages):
        # for each rock, use spec_crossing if older than the closest approach or 0 otherwise
        if age > T_closest_approach:
            spec_sh[n] = spec_crossing
        else:
            spec_sh[n] = 0
    return spec_sh


def main_runner():
    # start time (used to print time elapsed)
    t0 = time.time()

    # --------------------------------------
    # Part I: Initialize Parameters
    # --------------------------------------
    # start time (used to print time elapsed)
    t0 = time.time()
    # reference cross-section (in cm^2) to calculate DM spectra with
    ref_SIDD = 1e-46

    # ----------
    # Open runfile and store parameters
    if len(sys.argv) < 2:
        print("Usage: python3 subhalo_reach.py [runfile]")
        print("Exiting...")
        exit()

    # get run parameter filename
    fin_params = sys.argv[1]
    run_params = importlib.import_module(fin_params)

    # array of rock ages [Myr]
    ages = run_params.Youngest_sample_age + np.linspace(
        0,
        run_params.Sample_age_spacing * (run_params.N_samples - 1),
        run_params.N_samples,
    )

    # Likelihood setup
    try:
        Gaussian_likelihood = run_params.Gaussian_likelihood
    except:
        Gaussian_likelihood = False

    try:
        rel_bkg_sys = run_params.Relative_background_systematics
    except:
        rel_bkg_sys = 0.0

    # array of uranium concentration  [g/g]
    uranium_conc = np.full(run_params.N_samples, run_params.U_concentration)
    # masses of rocks [kg]
    sample_mass = np.full(run_params.N_samples, run_params.Sample_mass)

    # Build array theta_bool characterizing which optimization parameters are active
    #     - True indicates parameter is active, while False indicates inactive
    #     - Interpetations of each parameter listed below

    theta_bool = np.array(
        [
            run_params.Atmos_neutrino_background
        ]  # atmospheric neutrino background normalization
        + [run_params.DSNB_background]  # DSNB background normalization
        + [run_params.GSNB_background]  # GSNB background normalization
        + [
            run_params.Solar_neutrino_background
        ]  # solar neutrino background normalization
        + [True]
        * run_params.N_samples  # nrocks parameters for ages of rocks (always True)
        + [run_params.Optimize_mass]
        * run_params.N_samples  # nrocks parameters for masses of rocks
        + [run_params.Neutron_background or run_params.SingleAlpha_background]
        * run_params.N_samples  # nrocks parameters for uranium concentrations of rocks
        # (True if either neutron or single-alpha is turned on)
    )
    # number of active parameters
    # (not including WIMP cross section)
    nthetas = np.sum(theta_bool)

    # Initialize relative uncertainties on external constraints
    #     - Initial interpretation of parameters in ext_unc same as theta_bool,
    #       but constraints on inactive parameters will be removed
    #     - If input for uncertainty is "none", uncertainty will momentarily be
    #       set to nan and then removed

    def read_unc(x):
        # shorthand function to handle "None" inputs
        if x is None:
            return np.nan
        return float(x)

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

    # Initialize parameters for spectrum generation
    # lower edge of first track length bin (default value for smear_and_bin used here)
    xmin = -1.0
    # upper edge of last track length bin
    xmax = 10000.0

    indep_range = np.geomspace(
        run_params.Lbound_indepV, run_params.Ubound_indepV, run_params.bins_indepV
    )
    # if impact paramter is depedent variable, read in parameters for its interpolation range
    bsh_lower = run_params.Lbound_impact_parameter
    bsh_upper = run_params.Ubound_impact_parameter
    bsh_bins = run_params.N_impact_parameter_bins
    bsh_range = np.geomspace(bsh_lower, bsh_upper, bsh_bins)
    # read in paramaters for fixed subhalo parameters
    # variable parameters may be given a value or just "None" (if they are given a value, that value will be ignored)
    sh_params = np.array(
        [
            run_params.WIMP_xsec,
            run_params.WIMP_m,
            run_params.Subhalo_m,
            run_params.Concen_param,
            None,
            None,
        ]
    )

    # WIMP masses for the interpolation (in GeV)
    wimp_masses = np.geomspace(0.035, 1000.0, run_params.N_WIMP_ms)

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
        if Gaussian_likelihood:
            print("Using Gaussian likelihood for the number of events per bin")
            if rel_bkg_sys > 0.0:
                print(
                    "Including a relative systematic error ",
                    rel_bkg_sys,
                    " of the backgrounds",
                )
        else:
            if rel_bkg_sys > 0.0:
                print(
                    "Relative systematic error declared in RUNFILE ignored because Poisson-likelihood used"
                )
        print("")
        print("Number of samples:", run_params.N_samples)
        print("Youngest sample age (Myr):", run_params.Youngest_sample_age)
        print("Sample age spacing (Myr):", run_params.Sample_age_spacing)
        print("Time since closest approach (Myr):", run_params.T_closest_approach)
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
        if not (
            np.all(theta_bool[:4])
            and run_params.Optimize_mass
            and run_params.Neutron_background
            and run_params.SingleAlpha_background
        ):
            print("")
        # print uncertainties (or lack thereof) for constraints on active parameters
        for i, index in enumerate(
            [0, 1, 2, 3, 4, 4 + run_params.N_samples, 4 + 2 * run_params.N_samples]
        ):
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
        print("Hydrogen tracks included in spectra:", run_params.H_tracks)
        if run_params.Replace_subhalo_spectrum_with_halo:
            print("WARNING: Subhalo spectrum is set to halo value!")
        print("")
        print(
            "Impact parameter (pc) range for interpolation: ("
            + str(bsh_lower)
            + ", "
            + str(bsh_upper)
            + ")"
        )
        print("Number of impact parameter bins:", bsh_bins)
        print("Number of WIMP masses (for interpolation):", run_params.N_WIMP_ms)
        # print values for all fixed parameters
        sh_param_names = [
            "WIMP cross-section",
            "WIMP mass",
            "Subhalo mass",
            "Concentration parameter",
            "Impact parameter",
            "Relative velocity",
        ]
        for i in range(len(sh_param_names)):
            print(sh_param_names[i] + ": " + str(sh_params[i]))
        print("")
        sys.stdout.flush()

    # ----------
    # remove constraints on inactive parameters
    ext_unc = ext_unc[theta_bool]
    # boolean array of constraints for which uncertainties were given
    ext_bool = ~np.isnan(ext_unc)
    # remove constraints for which no uncertainties were given
    ext_unc = ext_unc[ext_bool]
    # divide out given cross-section by reference value
    sh_params[0] /= ref_SIDD

    # --------------------------------------
    # Part II: Compute Spectra
    # --------------------------------------
    # Compute neutrino background spectra
    #     - Each background spectrum has shape (nrocks, run_params.N_track_bins)
    #     - Spectra of all active neutrino backgrounds will be stacked into
    #       array spec_nu

    # initialize spectrum calculator
    calculator = CalcSpectra.CalcSpectra(
        run_params.Mineral, switch_keep_H=run_params.H_tracks
    )
    spec_nu = []  # array of neutrino background spectra

    # atmospheric neutrino spectrum
    if theta_bool[0]:
        spec_atm = (
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
            )  # track length spectrum per unit track length, unit time, and unit mass
            * ages[:, None]  # factor for age of rock
            * sample_mass[:, None]  # factor for mass of rock
        )
        spec_nu.append(spec_atm)

    # DSNB spectrum
    if theta_bool[1]:
        spec_DSNB = (
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
        spec_nu.append(spec_DSNB)

    # GSNB spectrum
    if theta_bool[2]:
        spec_GSNB = (
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
        spec_nu.append(spec_GSNB)

    # solar neutrino spectrum
    if theta_bool[3]:
        spec_solar = (
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
        spec_nu.append(spec_solar)

    # get number of neutrino backgrounds
    n_nu = len(spec_nu)
    # collect neutrino background spectra
    spec_nu = np.array(spec_nu)

    # ----------
    # Compute radiogenic neutron and single alpha-background spectra

    # neutron spectrum
    if run_params.Neutron_background:
        spec_neutrons = (
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
        spec_neutrons = np.zeros((run_params.N_samples, run_params.N_track_bins))


    # single-alpha background spectrum
    if run_params.SingleAlpha_background:
        spec_1a = (
            np.vstack(
                [
                    calculator.smear_and_bin_1a(
                        1.0,
                        run_params.Resolution,
                        xmin=xmin,
                        xmax=xmax,
                        nbins=run_params.N_track_bins,
                        logbins=True,
                    )[1]
                ]
                * run_params.N_samples
            )
            * uranium_conc[:, None]
            * sample_mass[:, None]
        )
    else:
        spec_1a = np.zeros((run_params.N_samples, run_params.N_track_bins))

    # ----------
    # Compute DM halo spectra (with parallelization)
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
            spec_halo_flux = p.map(func_halo, wimp_masses)
            p.close()
            p.join()
    else:
        spec_halo_flux = [func_halo(mass) for mass in wimp_masses]

    # rearrange result to the right form and normalize appropriately
    spec_halo_flux = np.array(spec_halo_flux)
    spec_halo_raw = spec_halo_flux[:, None, :] * ages[:, None] * sample_mass[:, None]
    # interpolate between WIMP masses
    # note that the interpolation returns the spectrum at the smallest (largest)
    # WIMP mass in wimp_masses if one calls spec_halo_interp(mass) for a mass
    # outside of the range of wimp_masses. Since this will yield a flat
    # contribution to the likelihood if the optimizer moves outside of the mass
    # range of halo_masses, it should force it to turn back.
    spec_halo_interp = interp1d(
        wimp_masses,
        spec_halo_raw,
        axis=0,
        bounds_error=False,
        fill_value=(spec_halo_raw[0], spec_halo_raw[-1]),
    )

    # ----------
    # Compute subhalo spectra (with parallelization)

    # generate list of inputs for get_spec_sh, based on the ranges
    # of the dependent (if impact parameter is the dependent variable)
    # and independent variables
    spec_sh_inputs = []
    # WIMP_xsec, WIMP_mass, Msh, csh, bsh, vsh
    for bsh in bsh_range:
        for indep_vals in indep_range:
            spec_sh_params = np.copy(sh_params)
            spec_sh_params[4] = bsh
            spec_sh_params[5] = indep_vals
            spec_sh_params = np.delete(spec_sh_params, 0)
            spec_sh_inputs.append(spec_sh_params)

    func_subhalo = partial(
        get_spec_sh,
        run_params.Mineral,
        run_params.H_tracks,
        ref_SIDD,
        run_params.Resolution,
        xmin,
        xmax,
        run_params.N_track_bins,
        run_params.T_closest_approach,
        ages,
        run_params.Sample_age_spacing,
        Replace_subhalo_spectrum_with_halo=run_params.Replace_subhalo_spectrum_with_halo,
    )

    # # compute get_spec_sh for all such inputs (using parallelization)
    if run_params.Cores > 1:
        with Pool(run_params.Cores) as p:
            spec_sh_flux = p.map(func_subhalo, spec_sh_inputs)
            p.close()
            p.join()

    else:
        spec_sh_flux = [func_subhalo(x) for x in spec_sh_inputs]

    # multiply by sample masses and appropriately reshape the resulting array
    # if impact parameter is the dependent variable, interpolate along impact parameter axis
    spec_sh_flux = np.array(spec_sh_flux)
    spec_sh_raw = spec_sh_flux * sample_mass[:, None]
    spec_sh_raw = spec_sh_raw.reshape(
        (
            len(bsh_range),
            len(indep_range),
            run_params.N_samples,
            run_params.N_track_bins,
        )
    )
    spec_sh_interp = interp1d(bsh_range, spec_sh_raw, axis=0)
    # 4 dimensions

    print("Spectra ready:", time.time() - t0, "s")
    sys.stdout.flush()

    # ----------
    # Compute reach for each WIMP mass (with parallelization)
    # and write results to file

    # function to apply correct reach function with all fixed parameters and
    # independent variables, given only values of independent variables
    # def get_reach(indep_vals):
    #     reach_params = np.copy(sh_params)
    #     for i, var_index in enumerate(indep_var):
    #         reach_params[var_index] = indep_vals[i]
    #     reach_params = np.delete(reach_params, dep_var)
    #     return get_reach_bsh(*reach_params)

    # norm_sh is just the WIMP cross section / ref_SIDD
    norm_sh = run_params.WIMP_xsec / ref_SIDD
    reach_sh_partial = partial(
        get_reach_bsh,
        spec_sh_interp,
        norm_sh,
        bsh_range,
        run_params.N_samples,
        n_nu,
        nthetas,
        ext_bool,
        ext_unc,
        run_params.WIMP_m,
        spec_halo_interp,
        spec_1a,
        spec_neutrons,
        spec_nu,
        indep_range,
        wimp_masses,
        include_bkg_nu_solar=run_params.Solar_neutrino_background,
        include_bkg_nu_GSNB=run_params.GSNB_background,
        include_bkg_nu_DSNB=run_params.DSNB_background,
        include_bkg_nu_atm=run_params.Atmos_neutrino_background,
        include_bkg_rad_1a=run_params.SingleAlpha_background,
        include_bkg_rad_neutrons=run_params.Neutron_background,
        Gaussian_likelihood=run_params.Gaussian_likelihood,
        Relative_background_systematics=run_params.Relative_background_systematics,
        mass_active=run_params.Optimize_mass,
    )

    if run_params.Cores > 1:
        with Pool(run_params.Cores) as p:
            reach = p.map(reach_sh_partial, indep_range)
            p.close()
            p.join()
    else:
        reach = [reach_sh_partial(indep_vals) for indep_vals in indep_range]
    reach = np.array(reach)
    print("Optimization finished:", time.time() - t0, "s")

    output = np.vstack((indep_range, reach)).T
    outfile = run_params.Output  # output filename
    np.savetxt(
        outfile,
        output,
        header="Relative velocity [km/s], Impact Parameter [pc]",
    )
    print("Results written to file:", time.time() - t0, "s")
    print("")
    print("Computation complete!")


if __name__ == "__main__":
    main_runner()

# def get_indep_index(params):
#     """
#     Shorthand function to find index in indep_range of given parameter choices
#     Input params should be array of [norm_sh, mass, Msh, csh, bsh, vsh]
#     """
#     # pull out parameters corresponding to independent variables
#     indep_params = np.array(params)[indep_var]
#     # identify index in indep_range which matches these parameters
#     indep_index = np.where([
#         np.all(indep_vals == indep_params) for indep_vals in indep_range
#     ])[0][0]
#     return indep_index

# some tiny number to be added to all spectra to avoid division by zero

# def get_reach_norm(mass, Msh, csh, bsh, vsh, threshold=1e14, no_reach_out_cm2=1e-31):
#     """
#     Function to calculate reach for cross-section
#     Searches for norm_sh which returns the critical
#     value of the test statistic
#     Inputs same as get_TS (except norm_sh)
#     kwargs:
#         - threshold: max normalization of log-likelihood allowed
#         - no_reach_out_cm2: value returned if the function
#                             finds no reach [cm^2]
#     """
#     # critical value of test statistic, given by Wilks' theorem
#     # (approximately 3.84)
#     TS_crit = chi2.ppf(0.95, 1)
#     # get smallest normalization from demanding that there
#     # must be at least 1 signal event in the oldest rock
#     indep_index = get_indep_index([np.nan, mass, Msh, csh, bsh, vsh])
#     l10_reach_min = np.log10(1.0 / np.sum(spec_sh_raw[indep_index, -1]))
#     # get max normalization from demanding that the
#     # log-likelihood of the true hypothesis is smaller than
#     # THRESHOLD to avoid numerical errors
#     l10_reach_max = optimize.brentq(
#         lambda l10_norm: (
#             np.sum(
#                 get_spec_asimov(10 ** l10_norm, mass, Msh, csh, bsh, vsh)
#                 * np.log(get_spec_asimov(10 ** l10_norm, mass, Msh, csh, bsh, vsh))
#                 - get_spec_asimov(10 ** l10_norm, mass, Msh, csh, bsh, vsh)
#             )
#             - threshold
#         ),
#         -100.0,
#         100.0,
#     )
#     # run root_finder
#     fun = lambda l10_norm: get_TS(10 ** l10_norm, mass, Msh, csh, bsh, vsh) - TS_crit
#     if fun(l10_reach_min) * fun(l10_reach_max) > 0:
#         l10_reach = np.log10(no_reach_out_cm2 / ref_SIDD)
#     else:
#         l10_reach = optimize.brentq(fun, l10_reach_min, l10_reach_max)
#     # print index in indep_range of these parameter choices
#     print(
#         "reach for params",
#         indep_index,
#         "of",
#         len(indep_range),
#         "computed.",
#         time.time() - t0,
#         "s",
#     )
#     sys.stdout.flush()
#     return 10 ** l10_reach