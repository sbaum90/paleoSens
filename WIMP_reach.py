import numpy as np
import sys
import multiprocessing as mp
import time
import importlib
from scipy import stats
from scipy import optimize
import iminuit
from functools import partial

from paleoSpec import CalcSpectra

# ---------------------------------------------
# Intro
# ---------------------------------------------
"""
Call as

python3 WIMP_reach.py RUNFILE

where RUNFILE is the name (without the .py file extension!) of the 
runfile. See WIMP_default_runfile.py for an example of the syntax
and the parameters which should be entered.

Per default, WIMP_reach uses a Poisson-likelihood to calculate the
sensitivity. It allows to incorporate external (Gaussian) constraints
on the nuissance parameters via the "ext_*" parameters in the RUNFILE.

If the optional parameter "Gaussian_likelihood" is set to 
"Gaussian_likelihood = True"
in the RUNFILE, the sensitivity is instead calculated by assuming
a normal distribution of the number of events in each bin. Per default, 
the code will compute the variance in each bin from the Poisson error
only. However, an additional parameters "rel_bkg_sys" can be included.
If this parameter is set to a value different from 0., the code will
project the sensitivity by including an additional RELATIVE systematic
error of the number of background events in each bin. Hence, in this case
the variance of the number of events in the i-th bin is set by
var_i = N_i + (rel_bkg_sys * N_i^bkg)**2
where N_i is the number of events in the ith bin N_i^bkg the number of
background events in the ith bin. Note that for the exclusion limit, 
N_i = N_i^bkg, while for the discovery reach, they differ by the 
contribution of the signal to the Asimov data.
Finally, note that if "Gaussian_likelihood != True" the code uses a
Poisson likelihood and the "rel_bkg_sys" parameter is ignored.
"""

# ---------------------------------------------
# some setup
# ---------------------------------------------
reach_max = 1e20 # some large number where we will stop the root finding
num_zero = 1e-100 # some small number added to spectra to avoid divions by zero

# ---------------------------------------------
# function to calculate DM spectrum
# ---------------------------------------------
def get_DM_spec(
    mineral_name,
    keep_H_tracks,
    readout_resolution_Aa,
    TR_xmin_Aa,
    TR_xmax_Aa,
    TR_nbins,
    TR_logbins,
    ref_xsec,
    DM_mass
    ):
    Spec_calculator = CalcSpectra.CalcSpectra(
        mineral_name, 
        switch_keep_H=keep_H_tracks
        )
    out = CalcSpectra.smear_and_bin(
        Spec_calculator.calc_dRdx_MW(DM_mass, ref_xsec),
        readout_resolution_Aa,
        xmin = TR_xmin_Aa,
        xmax = TR_xmax_Aa,
        nbins = int(TR_nbins),
        logbins = TR_logbins
        )[1]
    return out

# ---------------------------------------------
# functions for Asimov data sets
# ---------------------------------------------
# Asimov data for H0 (backgrounds only)
def calc_asimov_H0(Spec_bkgs, incl_bkg_bools):
    return np.sum(Spec_bkgs[incl_bkg_bools], axis=0)


# Asimov data for H1 (backgrounds + dark matter)
def calc_asimov_H1(Spec_bkgs, incl_bkg_bools, Spec_DM, xsec_relative):
    # note that the Spec_DM input should be a 1D array for a single DM mass
    return calc_asimov_H0(Spec_bkgs, incl_bkg_bools) + xsec_relative * Spec_DM


# ---------------------------------------------
# functions to build spectra for the two hypothesis as a
# function of the nuisance parameters
# ---------------------------------------------
def calc_spectrum_H0(
        thetas,
        TR_nbins,
        Spec_bkgs,
        incl_bkg_bools
        ):
    """
    Returns the expected spectrum for H0 (backgrounds only)
    needed for the likelihood. 
    inputs:
    - thetas - array of nuisance parameters:
        - thetas[0] - sample_age
        - thetas[1] - sample_mass
        - thetas[2:-1] - normalization for each included neutrino background
        - thetas[-1] - C238 if at least one radiogenic background is included 
    - see main_runner for the definition of all other input parameters
    """
    spec = np.zeros(int(TR_nbins))
    theta_ind = 2 # this is the index of the first normalization parameter in thetas
    if incl_bkg_bools[0]:
        spec += thetas[0] * thetas[1] * thetas[theta_ind] * Spec_bkgs[0]
        theta_ind += 1
    if incl_bkg_bools[1]:
        spec += thetas[0] * thetas[1] * thetas[theta_ind] * Spec_bkgs[1]
        theta_ind += 1
    if incl_bkg_bools[2]:
        spec += thetas[0] * thetas[1] * thetas[theta_ind] * Spec_bkgs[2]
        theta_ind += 1
    if incl_bkg_bools[3]:
        spec += thetas[0] * thetas[1] * thetas[theta_ind] * Spec_bkgs[3]
        theta_ind += 1
    if incl_bkg_bools[4]:
        spec += thetas[1] * thetas[theta_ind] * Spec_bkgs[4]
    if incl_bkg_bools[5]:
        spec += thetas[0] * thetas[1] * thetas[theta_ind] * Spec_bkgs[5]
    return spec


def calc_spectrum_H1(
    thetas,
    TR_nbins,
    Spec_bkgs,
    incl_bkg_bools,
    Spec_DM,
    xsec_relative
    ):
    """
    Returns the expected spectrum for H1 (background + signal)
    needed for the likelihood. 
    - thetas - array of nuisance parameters:
        - thetas[0] - sample_age
        - thetas[1] - sample_mass
        - thetas[2:-1] - normalization for each included neutrino background
        - thetas[-1] - C238 if at least one radiogenic background is included 
    - see main_runner for the definition of all other input parameter
    - note that the Spec_DM input should be a 1D array for a single DM mass
    """
    spec = calc_spectrum_H0(
        thetas,
        TR_nbins,
        Spec_bkgs,
        incl_bkg_bools
        )
    spec += thetas[0] * thetas[1] * xsec_relative * Spec_DM
    return spec


# ---------------------------------------------
# likelihood functions
# ---------------------------------------------
def calc_Poisson_log_lik(Asimov_data, Model_spectrum):
    """
    Returns the Poisson log likelihood for the Model_spectrum
    given the Asimov_data
    """
    return np.sum(Asimov_data * np.log(Model_spectrum) - Model_spectrum)


def calc_Gauss_constraints_log_lik(thetas, ext_unc, ext_bools):
    """
    returns the likelihood of the Gaussian constraints for the values 
    of the nuisance parameters thetas
    """
    return -0.5 * np.sum((((thetas - 1.) / ext_unc)[ext_bools])**2)


def calc_Gauss_log_lik(
    Asimov_data, 
    Model_spectrum, 
    Asimov_data_bkg, 
    rel_bkg_sys=0.
    ):
    """
    Returns the Gaussian log likelihood for the Model_spectrum
    given the Asimov_data including an extra relative systematic
    error to the variance given by the 
    (rel_bkg_sys * Asimov_data_bkg)**2 per bin
    """
    out = -0.5 * np.sum(
        (Asimov_data - Model_spectrum)**2
        / (Asimov_data + (rel_bkg_sys * Asimov_data_bkg)**2)     
        )
    return out


def TS_func_exclusion(
    TR_nbins,
    Gaussian_likelihood,
    rel_bkg_sys,
    use_minimizer_minuit,
    use_minimizer_powell,
    N_thetas,
    bnds,
    ext_unc, 
    ext_bools,
    Spec_bkgs, 
    incl_bkg_bools,
    Spec_DM,
    xsec_relative
    ):
    # see main_runner for the definition of input parameter
    # - note that the Spec_DM input should be a 1D array for a single DM mass
    # generate Asimov data
    Asimov_data = calc_asimov_H0(Spec_bkgs, incl_bkg_bools)+num_zero
    # define function to optimize the nuisance parameters over
    if not Gaussian_likelihood:
        fun = (
            lambda thetas: 2.*(
                calc_Poisson_log_lik(Asimov_data, Asimov_data)
                - (
                    calc_Poisson_log_lik(
                        Asimov_data,
                        ( 
                            calc_spectrum_H1(
                                thetas,
                                TR_nbins,
                                Spec_bkgs,
                                incl_bkg_bools,
                                Spec_DM,
                                xsec_relative
                                ) 
                            + num_zero
                            )
                        )
                    + calc_Gauss_constraints_log_lik(thetas, ext_unc, ext_bools)
                    )
                )
            )
    else:
        fun = (
            lambda thetas: 2.*(
                - calc_Gauss_log_lik(
                    Asimov_data,
                    (
                        calc_spectrum_H1(
                            thetas,
                            TR_nbins,
                            Spec_bkgs,
                            incl_bkg_bools,
                            Spec_DM,
                            xsec_relative
                            )
                        + num_zero
                        ),
                    Asimov_data,
                    rel_bkg_sys = rel_bkg_sys
                    )
                - calc_Gauss_constraints_log_lik(thetas, ext_unc, ext_bools)
                )
            )
    if use_minimizer_minuit:
        # set up iminuit computation
        optimizer = iminuit.Minuit(fun, np.ones(N_thetas))
        optimizer.limits = bnds
        optimizer.errordef = optimizer.LIKELIHOOD
        # optimize
        optimizer.migrad()
        # get result
        TS = optimizer.fval
    elif use_minimizer_powell:
        optimizer_output = optimize.minimize(
            fun,
            np.ones(N_thetas), 
            method = 'Powell',
            bounds=bnds
            )
        TS = optimizer_output.fun
    return TS


def TS_func_discovery(
    TR_nbins,
    Gaussian_likelihood,
    rel_bkg_sys,
    use_minimizer_minuit,
    use_minimizer_powell,
    N_thetas,
    bnds,
    ext_unc, 
    ext_bools,
    Spec_bkgs, 
    incl_bkg_bools,
    Spec_DM,
    xsec_relative
    ):
    # see main_runner for the definition of input parameter
    # - note that the Spec_DM input should be a 1D array for a single DM mass
    # generate Asimov data
    Asimov_data = (
        calc_asimov_H1(
            Spec_bkgs, 
            incl_bkg_bools, 
            Spec_DM, 
            xsec_relative
            )
        + num_zero
        )
    # define function to optimize the nuisance parameters over
    if not Gaussian_likelihood:
        fun = (
            lambda thetas: 2.*(
                calc_Poisson_log_lik(Asimov_data, Asimov_data)
                - (
                    calc_Poisson_log_lik(
                        Asimov_data, 
                        (
                            calc_spectrum_H0(
                                thetas,
                                TR_nbins,
                                Spec_bkgs,
                                incl_bkg_bools
                                ) 
                            + num_zero
                            )
                        )
                    + calc_Gauss_constraints_log_lik(thetas, ext_unc, ext_bools)
                    )
                )
            )
    else:
        Asimov_data_bkg = calc_asimov_H0(Spec_bkgs, incl_bkg_bools)+num_zero
        fun = (
            lambda thetas: 2.*(
                - calc_Gauss_log_lik(
                    Asimov_data,
                    ( 
                        calc_spectrum_H0(
                            thetas,
                            TR_nbins,
                            Spec_bkgs,
                            incl_bkg_bools
                            )
                        + num_zero
                        ),
                    Asimov_data_bkg,
                    rel_bkg_sys = rel_bkg_sys
                    )
                - calc_Gauss_constraints_log_lik(thetas, ext_unc, ext_bools)
                )
            )
    if use_minimizer_minuit:
        # set up iminuit computation
        optimizer = iminuit.Minuit(fun, np.ones(N_thetas))
        optimizer.limits = bnds
        optimizer.errordef = optimizer.LIKELIHOOD
        # optimize
        optimizer.migrad()
        # get result
        TS = optimizer.fval
    elif use_minimizer_powell:
        optimizer_output = optimize.minimize(
            fun,
            np.ones(N_thetas), 
            method = 'Powell',
            bounds=bnds
            )
        TS = optimizer_output.fun
    return TS

# ---------------------------------------------
# functions to compute reach
# ---------------------------------------------

# projected exclusion limit
def get_exclusion_reach(
    TR_nbins,
    Gaussian_likelihood,
    rel_bkg_sys,
    use_minimizer_minuit,
    use_minimizer_powell,
    N_thetas,
    bnds,
    ext_unc, 
    ext_bools,
    Spec_bkgs, 
    incl_bkg_bools,
    TS_threshold_exclusion,
    Spec_DM
    ):
    # see main_runner for the definition of input parameter
    #- note that the Spec_DM input should be a 1D array for a single DM mass
    # get smallest allowed normalization from demanding that there
    # must be at least one signal event
    if np.sum(Spec_DM) > 0:
        reach_min = 1. / np.sum(Spec_DM)
    else:
        return reach_max
    # run root finder
    fun = (
        lambda xsec_relative: TS_func_exclusion(
            TR_nbins,
            Gaussian_likelihood,
            rel_bkg_sys,
            use_minimizer_minuit,
            use_minimizer_powell,
            N_thetas,
            bnds,
            ext_unc, 
            ext_bools,
            Spec_bkgs, 
            incl_bkg_bools,
            Spec_DM,
            xsec_relative
            )
        - TS_threshold_exclusion
        )
    if fun(reach_min) > 0:
        reach = reach_min
    else:
        reach, r = optimize.brentq(
            fun,
            reach_min, 
            reach_max,
            full_output = True,
            disp = False
            )
        if not r.converged:
            reach = reach_max
    return reach


# projected discovery reach
def get_discovery_reach(
    TR_nbins,
    Gaussian_likelihood,
    rel_bkg_sys,
    use_minimizer_minuit,
    use_minimizer_powell,
    N_thetas,
    bnds,
    ext_unc, 
    ext_bools,
    Spec_bkgs, 
    incl_bkg_bools,
    TS_threshold_discovery,
    Spec_DM
    ):
    # get smallest allowed normalization from demanding that there
    # must be at least one signal event
    if np.sum(Spec_DM) > 0:
        reach_min = 1. / np.sum(Spec_DM)
    else:
        return reach_max
    # run root finder
    fun = (
        lambda xsec_relative: TS_func_discovery(
            TR_nbins,
            Gaussian_likelihood,
            rel_bkg_sys,
            use_minimizer_minuit,
            use_minimizer_powell,
            N_thetas,
            bnds,
            ext_unc, 
            ext_bools,
            Spec_bkgs, 
            incl_bkg_bools,
            Spec_DM,
            xsec_relative
            )
        - TS_threshold_discovery
        )
    if fun(reach_min) > 0:
        reach = reach_min
    else:
        reach, r = optimize.brentq(
            fun,
            reach_min,
            reach_max,
            full_output = True,
            disp = False
            )
        if not r.converged:
            reach = reach_max
    return reach


# ---------------------------------------------
# main wrapper function
# ---------------------------------------------
def main_runner():
    start_t = time.time() # start time for messages
    # ---------------------------------------------
    # declare extra parameters not declared in RUNFILE
    # ---------------------------------------------
    TS_threshold_exclusion = stats.chi2.ppf(0.90, 1)
    CL_discovery = stats.chi2.cdf(5.**2, 1) # CL corresponding to 5 sigma for 1 dof
    TS_threshold_discovery = stats.chi2.ppf(CL_discovery, 1)
    # ---------------------------------------------
    # choose minimizer (hardcoded)
    # ---------------------------------------------
    use_minimizer_minuit = True
    use_minimizer_powell = False
    if not (use_minimizer_minuit ^ use_minimizer_powell):
        print("Inconsistent choice of minimizer")
        print("Only one option should be set to True")
        print("Exiting...")
        sys.exit()
    # ---------------------------------------------
    # import parameters from RUNFILE
    # ---------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python3 WIMP_reach.py [runfile]")
        print("Exiting...")
        sys.exit()
    fin_params = sys.argv[1]
    run_params = importlib.import_module(fin_params)
    # combine parameters into arrays
    ext_bools = np.array([
        run_params.ext_sample_age_bool,
        run_params.ext_sample_mass_bool, 
        run_params.ext_nu_solar_bool, 
        run_params.ext_nu_GSNB_bool, 
        run_params.ext_nu_DSNB_bool, 
        run_params.ext_nu_atm_bool, 
        run_params.ext_C238_bool
        ])
    #
    ext_unc = np.array([
        run_params.ext_sample_age_unc,
        run_params.ext_sample_mass_unc, 
        run_params.ext_nu_solar_unc, 
        run_params.ext_nu_GSNB_unc, 
        run_params.ext_nu_DSNB_unc, 
        run_params.ext_nu_atm_unc, 
        run_params.ext_C238_unc
        ])
    #
    incl_bkg_bools = np.array([
        run_params.include_bkg_nu_solar, 
        run_params.include_bkg_nu_GSNB, 
        run_params.include_bkg_nu_DSNB, 
        run_params.include_bkg_nu_atm, 
        run_params.include_bkg_rad_1a, 
        run_params.include_bkg_rad_neutrons
        ])
    # check if Gaussian_likelihood and rel_bkg_sys are declared
    try:
        run_params.Gaussian_likelihood
    except:
        run_params.Gaussian_likelihood = False
    #
    try:
        run_params.rel_bkg_sys
    except:
        run_params.rel_bkg_sys = 0.
    # ---------------------------------------------
    # print some global info to std.out
    # ------------------------------------------
    if run_params.verbose:
        print("#----------------------------------")
        print("Sucessfully imported parameters")
        print("\nResults will be stored in: ")
        print(run_params.fout_name+"_"+run_params.mineral_name+"_*.txt")
        print("")
        print("#----------------------------------")
        print("")
        print("age of the target sample [Myr]: ", run_params.sample_age_Myr)
        print("mass of the target sample in [kg]: ", run_params.sample_mass_kg) 
        print("track length resolution in [Å]: ", run_params.readout_resolution_Aa)
        print("uranium-238 concentration in [g/g]: ", run_params.C238) 
        print("target mineral: ", run_params.mineral_name)
        print("including H tracks?: ", run_params.keep_H_tracks)
        print("")
        print("#----------------------------------")
        print("")
        print("including external constraint on sample age? ", run_params.ext_sample_age_bool)
        if run_params.ext_sample_age_bool:
            print("relative uncertainty: ", run_params.ext_sample_age_unc)
        print("including external constraint on sample mass? ", run_params.ext_sample_mass_bool)
        if run_params.ext_sample_mass_bool:
            print("relative uncertainty: ", run_params.ext_sample_mass_unc)
        print("including external constraint on solar nu flux? ", run_params.ext_nu_solar_bool)
        if run_params.ext_nu_solar_bool:
            print("relative uncertainty: ", run_params.ext_nu_solar_unc)
        print("including external constraint on GSNB normalization? ", run_params.ext_nu_GSNB_bool)
        if run_params.ext_nu_GSNB_bool:
            print("relative uncertainty: ", run_params.ext_nu_GSNB_unc)
        print("including external constraint on DSNB normalization? ", run_params.ext_nu_DSNB_bool)
        if run_params.ext_nu_DSNB_bool:
            print("relative uncertainty: ", run_params.ext_nu_DSNB_unc)
        print("including external constraint on atmospheric nu flux? ", run_params.ext_nu_atm_bool)
        if run_params.ext_nu_atm_bool:
            print("relative uncertainty: ", run_params.ext_nu_atm_unc)
        print("including external constraint on uranium-238 concentration? ", run_params.ext_C238_bool)
        if run_params.ext_C238_bool:
            print("relative uncertainty: ", run_params.ext_C238_unc)
        print("")
        print("#----------------------------------")
        print("")
        print("lower edge of smallest track length bin in [Aa]: ", run_params.TR_xmin_Aa)
        print("  if ==-1, the code uses readout_resolution/2")
        print("upper edge of the largest track length bin in [Aa]: ", run_params.TR_xmax_Aa)
        print("  Should not be chosen larger than 10,000")
        print("using log-spaced track length bins: ", run_params.TR_logbins)  
        print("number of track-length bins: ", run_params.TR_nbins)
        print("  If TR_logbins == False, this can be set to -1,")
        print("  in which case the bin-width is set to readout_resolution/2")
        print("")
        print("#----------------------------------")
        print("")
        print("smallest DM mass in [GeV] for which the limit is computed: ", run_params.DMmass_min_GeV)
        print("largest DM mass in [GeV] for which the limit is computed: ", run_params.DMmass_max_GeV)   
        print("number of (log-spaced) mass bins: ", run_params.DMmass_nbins)   
        print("")
        print()
        if run_params.Gaussian_likelihood:
            print("Using Gaussian likelihood for the number of events per bin")
            if run_params.rel_bkg_sys > 0.:
                print("Including a relative systematic error ", run_params.rel_bkg_sys, " of the backgrounds")
        else:
            if run_params.rel_bkg_sys > 0:
                print("Relative systematic error declared in RUNFILE ignored because Gaussian_likelihood != True")
        print("computing projected 90% exclusion limit: ", run_params.output_exclusion_sens)
        print("computing projected 5-sigma discovery limit: ", run_params.output_discovery_sens)
        print("")
        print("#----------------------------------")
        print("")
        print("number of cores used for parallelization: ", run_params.Ncores_mp)
        print("")
        if not run_params.include_bkg_nu_solar:
            print("WARNING: solar neutrino background is turned off")
            print("  This should be used for testing only")
            print("")
        if not run_params.include_bkg_nu_GSNB:
            print("WARNING: galactic supernova neutrino background is turned off")
            print("  This should be used for testing only")
            print("")
        if not run_params.include_bkg_nu_DSNB:
            print("WARNING: diffuse supernova neutrino background is turned off")
            print("  This should be used for testing only")
            print("")
        if not run_params.include_bkg_nu_atm:
            print("WARNING: atmospheric background is turned off")
            print("  This should be used for testing only")
            print("")
        if not run_params.include_bkg_rad_1a:
            print("WARNING: radiogenic single-alpha background is turned off")
            print("  This should be used for testing only")
            print("")
        if not run_params.include_bkg_rad_neutrons:
            print("WARNING: radiogenic neutron background is turned off")
            print("  This should be used for testing only")
            print("")
        print("#----------------------------------")
    # ---------------------------------------------
    # generate the spectra
    # ---------------------------------------------
    if run_params.verbose:
        print("")
        print("Starting to generate the spectra.")
        print("Elapsed time: " + str(time.time() - start_t) + " s")
    if run_params.mineral_name not in CalcSpectra.good_mineral_list:
        print("you asked for the target mineral: ", run_params.mineral_name)
        print("this program regrets it doesn't know this mineral")
        print("the known minerals are:")
        for mineral in CalcSpectra.good_mineral_list:
            print("  ", mineral)
        print("Exiting...")
        sys.exit()
    # initialize spectrum generator
    Spec_calculator = CalcSpectra.CalcSpectra(run_params.mineral_name, switch_keep_H=run_params.keep_H_tracks)
    ref_xsec = 1e-46 # reference cross sections for which DM signal are calculated
    # background spectra
    Spec_nu_solar = (
        CalcSpectra.smear_and_bin(
            Spec_calculator.calc_dRdx_BkgNeu_solar(),
            run_params.readout_resolution_Aa,
            xmin = run_params.TR_xmin_Aa,
            xmax = run_params.TR_xmax_Aa,
            nbins = int(run_params.TR_nbins),
            logbins = run_params.TR_logbins
            )[1]
        * run_params.sample_age_Myr
        * run_params.sample_mass_kg
        )
    #
    Spec_nu_GSNB = (
        CalcSpectra.smear_and_bin(
            Spec_calculator.calc_dRdx_BkgNeu_GSNB(),
            run_params.readout_resolution_Aa,
            xmin = run_params.TR_xmin_Aa,
            xmax = run_params.TR_xmax_Aa,
            nbins = int(run_params.TR_nbins),
            logbins = run_params.TR_logbins
            )[1]
        * run_params.sample_age_Myr
        * run_params.sample_mass_kg
        )
    #
    Spec_nu_DSNB = (
        CalcSpectra.smear_and_bin(
            Spec_calculator.calc_dRdx_BkgNeu_DSNB(),
            run_params.readout_resolution_Aa,
            xmin = run_params.TR_xmin_Aa,
            xmax = run_params.TR_xmax_Aa,
            nbins = int(run_params.TR_nbins),
            logbins = run_params.TR_logbins
            )[1]
        * run_params.sample_age_Myr
        * run_params.sample_mass_kg
        )
    #
    Spec_nu_atm = (
        CalcSpectra.smear_and_bin(
            Spec_calculator.calc_dRdx_BkgNeu_atm(),
            run_params.readout_resolution_Aa,
            xmin = run_params.TR_xmin_Aa,
            xmax = run_params.TR_xmax_Aa,
            nbins = int(run_params.TR_nbins),
            logbins = run_params.TR_logbins
            )[1]
        * run_params.sample_age_Myr
        * run_params.sample_mass_kg
        )
    #
    Spec_rad_1a = (
        Spec_calculator.smear_and_bin_1a(
            run_params.C238, 
            run_params.readout_resolution_Aa, 
            xmin = run_params.TR_xmin_Aa,
            xmax = run_params.TR_xmax_Aa,
            nbins = int(run_params.TR_nbins),
            logbins = run_params.TR_logbins
            )[1]
        * run_params.sample_mass_kg
        )
    #
    Spec_rad_neutrons = (
        CalcSpectra.smear_and_bin(
            Spec_calculator.calc_dRdx_Bkgn(run_params.C238),
            run_params.readout_resolution_Aa,
            xmin = run_params.TR_xmin_Aa,
            xmax = run_params.TR_xmax_Aa,
            nbins = int(run_params.TR_nbins),
            logbins = run_params.TR_logbins
            )[1]
        * run_params.sample_age_Myr
        * run_params.sample_mass_kg
        )
    #
    # build list of all backgrounds in same order as incl_bkg_bools
    Spec_bkgs = np.array([
        Spec_nu_solar,
        Spec_nu_GSNB,
        Spec_nu_DSNB,
        Spec_nu_atm,
        Spec_rad_1a,
        Spec_rad_neutrons
        ])
    if run_params.verbose:
        print("")
        print("#----------------------------------")
        print("")
        print("Done generating the background spectra...")
        print("Time: " + str(time.time() - start_t) + " s")
    # Dark matter spectrum; parallelized
    DM_masses = np.geomspace(run_params.DMmass_min_GeV, 
        run_params.DMmass_max_GeV,
        int(run_params.DMmass_nbins))
    fun = partial(
        get_DM_spec,
        run_params.mineral_name,
        run_params.keep_H_tracks,
        run_params.readout_resolution_Aa,
        run_params.TR_xmin_Aa,
        run_params.TR_xmax_Aa,
        run_params.TR_nbins,
        run_params.TR_logbins,
        ref_xsec
        )
    if run_params.Ncores_mp > 1:
        pool = mp.Pool(run_params.Ncores_mp)
        Spec_DM = pool.map_async(fun, DM_masses).get()
        pool.close()
        pool.join()
    else:
        Spec_DM = [fun(m) for m in DM_masses]
    # convert to numpy array and normalize
    Spec_DM = run_params.sample_age_Myr * run_params.sample_mass_kg * np.array(Spec_DM)
    if run_params.verbose:
        print("")
        print("#----------------------------------")
        print("")
        print("Done generating the signal spectra...")
        print("Time: " + str(time.time() - start_t) + " s")
    # ---------------------------------------------
    # build log likelihood functions
    # ---------------------------------------------
    # need to cut the ext_ constraints to the right 
    # size to match the list of nuisance parameters,
    # which is ordered:
    #   thetas_nuisance[0] - sample_age
    #   thetas_nuisance[1] - sample_mass
    #   thetas_nuisance[2:-1] - normalization for each included neutrino background
    #   thetas_nuisance[-1] - C238 if at least one radiogenic background is included 
    ext_bools = np.delete(ext_bools, np.where(incl_bkg_bools[:4]==False)[0]+2)
    ext_unc = np.delete(ext_unc, np.where(incl_bkg_bools[:4]==False)[0]+2)
    if not incl_bkg_bools[4] and not incl_bkg_bools[5]:
        ext_bools = np.delete(ext_bools, -1)
        ext_unc = np.delete(ext_unc, -1)
    #
    N_thetas = 2 + sum(incl_bkg_bools[:4]) + (incl_bkg_bools[4] or incl_bkg_bools[5]) # number of nuisance parameters
    bnds = [(0, None)]*N_thetas # bounds for optimizer
    # ---------------------------------------------
    # compute the reach, parallelized
    # ---------------------------------------------
    if run_params.output_exclusion_sens:
        if run_params.verbose:
            print("")
            print("#----------------------------------")
            print("")
            print("Calculating the exclusion limit...")
            print("Time: " + str(time.time() - start_t) + " s")
            print("")
        fun = partial(
            get_exclusion_reach,
            run_params.TR_nbins,
            run_params.Gaussian_likelihood,
            run_params.rel_bkg_sys,
            use_minimizer_minuit,
            use_minimizer_powell,
            N_thetas,
            bnds,
            ext_unc, 
            ext_bools,
            Spec_bkgs, 
            incl_bkg_bools,
            TS_threshold_exclusion
            )
        if run_params.Ncores_mp > 1:
            pool = mp.Pool(run_params.Ncores_mp)
            reach = pool.map_async(fun, Spec_DM).get()
            pool.close()
            pool.join()
        else:
            reach = [fun(DM_spec) for DM_spec in Spec_DM]
        # write output to file
        output = np.array([DM_masses, ref_xsec*np.array(reach)]).T
        fout = run_params.fout_name + "_"+run_params.mineral_name+"_exclusion.txt"
        np.savetxt(fout, output, header="Mass [GeV], WIMP-nucleon cross section [cm^2]")
        if run_params.verbose:
            print("Finished.")
            print("Time: " + str(time.time() - start_t) + " s")
        del reach, output, fout
    #
    if run_params.output_discovery_sens:
        if run_params.verbose:
            print("")
            print("#----------------------------------")
            print("")
            print("Calculating the discovery reach...")
            print("Time: " + str(time.time() - start_t) + " s")
        fun = partial(
            get_discovery_reach,
            run_params.TR_nbins,
            run_params.Gaussian_likelihood,
            run_params.rel_bkg_sys,
            use_minimizer_minuit,
            use_minimizer_powell,
            N_thetas,
            bnds,
            ext_unc, 
            ext_bools,
            Spec_bkgs, 
            incl_bkg_bools,
            TS_threshold_discovery
            )
        if run_params.Ncores_mp > 1:
            pool = mp.Pool(run_params.Ncores_mp)
            reach = pool.map_async(fun, Spec_DM).get()
            pool.close()
            pool.join()
        else:
            reach = [fun(DM_spec) for DM_spec in Spec_DM]
        # write output to file
        output = np.array([DM_masses, ref_xsec*np.array(reach)]).T
        fout = run_params.fout_name + "_"+run_params.mineral_name+"_discovery.txt"
        np.savetxt(fout, output, header="Mass [GeV], WIMP-nucleon cross section [cm^2]")
        if run_params.verbose:
            print("Finished.")
            print("Time: " + str(time.time() - start_t) + " s")
        del reach, output, fout
    #
    if run_params.verbose:
        print("")
        print("#----------------------------------")


if __name__ == "__main__":
    main_runner()
