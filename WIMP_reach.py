import numpy as np
import sys
import multiprocessing as mp
import time
import importlib
from scipy import stats
from scipy import optimize
import iminuit

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
start_t = time.time() # start time for messages
num_zero = 1e-100 # some small number added to spectra to avoid divions by zero

# ---------------------------------------------
# extra parameters not declared in RUNFILE
# ---------------------------------------------
TS_threshold_exclusion = stats.chi2.ppf(0.90, 1)
CL_discovery = stats.chi2.cdf(5.**2, 1) # CL corresponding to 5 sigma for 1 dof
TS_threshold_discovery = stats.chi2.ppf(CL_discovery, 1)

# choose minimizer
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

ext_unc = np.array([
    run_params.ext_sample_age_unc,
    run_params.ext_sample_mass_unc, 
    run_params.ext_nu_solar_unc, 
    run_params.ext_nu_GSNB_unc, 
    run_params.ext_nu_DSNB_unc, 
    run_params.ext_nu_atm_unc, 
    run_params.ext_C238_unc
    ])

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

try:
    run_params.rel_bkg_sys
except:
    run_params.rel_bkg_sys = 0.


# print some info to std.out
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

def get_DM_spec(mass_ind):
    out = CalcSpectra.smear_and_bin(
        Spec_calculator.calc_dRdx_MW(DM_masses[mass_ind], ref_xsec),
        run_params.readout_resolution_Aa,
        xmin = run_params.TR_xmin_Aa,
        xmax = run_params.TR_xmax_Aa,
        nbins = int(run_params.TR_nbins),
        logbins = run_params.TR_logbins
        )[1]
    return out

if __name__ == '__main__' and run_params.Ncores_mp > 1:
    pool = mp.Pool(run_params.Ncores_mp)
    Spec_DM = pool.map_async(get_DM_spec, range(run_params.DMmass_nbins)).get()
    pool.close()
    pool.join()
else:
    Spec_DM = [get_DM_spec(mass_ind) for mass_ind in range(run_params.DMmass_nbins)]

# convert to numpy array and normalize
Spec_DM = run_params.sample_age_Myr * run_params.sample_mass_kg * np.array(Spec_DM)

if run_params.verbose:
    print("")
    print("#----------------------------------")
    print("")
    print("Done generating the signal spectra...")
    print("Time: " + str(time.time() - start_t) + " s")

# ---------------------------------------------
# build Asimov data sets
# ---------------------------------------------
# Asimov data for H0 (backgrounds only)
def calc_asimov_H0():
    return np.sum(Spec_bkgs[incl_bkg_bools], axis=0)


# Asimov data for H1 (backgrounds + dark matter)
def calc_asimov_H1(mass_ind, xsec_relative):
    return calc_asimov_H0() + xsec_relative*Spec_DM[mass_ind,:]


# ---------------------------------------------
# build Spectra for the two hypothesis as a
# function of the nuisance parameters
# ---------------------------------------------
def calc_spectrum_H0(thetas):
    """
    Returns the expected spectrum for H0 (backgrounds only)
    needed for the likelihood. 
    inputs:
    - thetas_nuisance - array of nuisance parameters:
        - thetas_nuisance[0] - sample_age
        - thetas_nuisance[1] - sample_mass
        - thetas_nuisance[2:-1] - normalization for each included neutrino background
        - thetas_nuisance[-1] - C238 if at least one radiogenic background is included 
    """
    spec = np.zeros(int(run_params.TR_nbins))
    theta_ind = 2 # this is the index of the first normalization parameter in thetas
    if run_params.include_bkg_nu_solar:
        spec += (
            thetas[0]
            *thetas[1]
            *thetas[theta_ind]
            *Spec_nu_solar
            )
        theta_ind += 1
    if run_params.include_bkg_nu_GSNB:
        spec += (
            thetas[0]
            *thetas[1]
            *thetas[theta_ind]
            *Spec_nu_GSNB
            )
        theta_ind += 1
    if run_params.include_bkg_nu_DSNB:
        spec += (
            thetas[0]
            *thetas[1]
            *thetas[theta_ind]
            *Spec_nu_DSNB
            )
        theta_ind += 1
    if run_params.include_bkg_nu_atm:
        spec += (
            thetas[0]
            *thetas[1]
            *thetas[theta_ind]
            *Spec_nu_atm
            )
        theta_ind += 1
    if run_params.include_bkg_rad_1a:
        spec += (
            thetas[1]
            *thetas[theta_ind]
            *Spec_rad_1a
            )
    if run_params.include_bkg_rad_neutrons:
        spec += (
            thetas[0]
            *thetas[1]
            *thetas[theta_ind]
            *Spec_rad_neutrons
            )
    return spec


def calc_spectrum_H1(mass_ind, xsec_relative, thetas_nuisance):
    """
    Returns the expected spectrum for H1 (background + signal)
    needed for the likelihood. 
    inputs:
    - mass inds - index of DM_masses
    - xsec_relative - DM cross section/xsec_ref
    - thetas_nuisance - array of nuisance
      parameters (see calc_spectrum_H0)
    """
    spec = calc_spectrum_H0(thetas_nuisance)
    spec += (
        thetas_nuisance[0]
        * thetas_nuisance[1]
        * xsec_relative
        * Spec_DM[mass_ind,:]
        )
    return spec

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


def calc_Poisson_log_lik(Asimov_data, Model_spectrum):
    """
    Returns the Poisson log likelihood for the Model_spectrum
    given the Asimov_data
    """
    return np.sum( Asimov_data * np.log(Model_spectrum) - Model_spectrum )


def calc_Gauss_constraints_log_lik(thetas_nuisance):
    """
    returns the likelihood of the Gaussian constraints for the values 
    of the nuisance parameters thetas_nuisance
    """
    return -0.5 * np.sum( ( ((thetas_nuisance - 1.)/ext_unc)[ext_bools] )**2 )


def calc_Gauss_log_lik(
    Asimov_data, 
    Model_spectrum, 
    Asimov_data_bkg, 
    rel_bkg_sys=run_params.rel_bkg_sys
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


# ---------------------------------------------
# build functions for maximization of the test
# statistic (TS = -2 * log likelihood ratio) 
# over the nuisance parameters
# ---------------------------------------------
# order of nuisance parameters:
#     thetas_nuisance[0] - sample_age
#     thetas_nuisance[1] - sample_mass
#     thetas_nuisance[2:-1] - normalization for each included neutrino background
#     thetas_nuisance[-1] - C238 if at least one radiogenic background is included

N_thetas = 2 + sum(incl_bkg_bools[:4]) + (incl_bkg_bools[4] or incl_bkg_bools[5]) # number of nuisance parameters
bnds = [(0, None)]*N_thetas # bounds for optimizer

def TS_func_exclusion(mass_ind, xsec_relative):
    # generate Asimov data
    Asimov_data = calc_asimov_H0()+num_zero
    # define function to optimize the nuisance parameters over
    if not run_params.Gaussian_likelihood:
        fun = lambda thetas: 2.*(
            calc_Poisson_log_lik(Asimov_data, Asimov_data)
            - (
                calc_Poisson_log_lik(
                    Asimov_data, 
                    calc_spectrum_H1(mass_ind, xsec_relative, thetas)+num_zero
                    )
                + calc_Gauss_constraints_log_lik(thetas)
                )
            )
    else:
        fun = lambda thetas: 2.*(
            - calc_Gauss_log_lik(
                Asimov_data, 
                calc_spectrum_H1(mass_ind, xsec_relative, thetas)+num_zero,
                Asimov_data
                )
            - calc_Gauss_constraints_log_lik(thetas)
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


def TS_func_discovery(mass_ind, xsec_relative):
    # generate Asimov data
    Asimov_data = calc_asimov_H1(mass_ind, xsec_relative)+num_zero
    # define function to optimize the nuisance parameters over
    if not run_params.Gaussian_likelihood:
        fun = lambda thetas: 2.*(
            calc_Poisson_log_lik(Asimov_data, Asimov_data)
            - (
                calc_Poisson_log_lik(Asimov_data, calc_spectrum_H0(thetas)+num_zero)
                + calc_Gauss_constraints_log_lik(thetas)
                )
            )
    else:
        Asimov_data_bkg = calc_asimov_H0()+num_zero
        fun = lambda thetas: 2.*(
            - calc_Gauss_log_lik(
                Asimov_data, 
                calc_spectrum_H0(thetas)+num_zero,
                Asimov_data_bkg
                )
            - calc_Gauss_constraints_log_lik(thetas)
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
# compute reach
# ---------------------------------------------

reach_max = 1e20 # some large number where we will stop the root finding

# projected exclusion limit; parallelized
def get_exclusion_reach(mass_ind):
    # get smallest allowed normalization from demanding that there
    # must be at least one signal event
    if np.sum(Spec_DM[mass_ind,:]) > 0:
        reach_min = 1. / np.sum(Spec_DM[mass_ind,:])
    else:
        return reach_max
    # run root finder
    fun = lambda xsec_relative: TS_func_exclusion(mass_ind, xsec_relative) - TS_threshold_exclusion
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


if run_params.output_exclusion_sens:
    if run_params.verbose:
        print("")
        print("#----------------------------------")
        print("")
        print("Calculating the exclusion limit...")
        print("Time: " + str(time.time() - start_t) + " s")
        print("")
    if __name__ == '__main__' and run_params.Ncores_mp > 1:
        pool = mp.Pool(run_params.Ncores_mp)
        reach = pool.map_async(get_exclusion_reach, range(run_params.DMmass_nbins)).get()
        pool.close()
        pool.join()
    else:
        reach = [get_exclusion_reach(mass_ind) for mass_ind in range(run_params.DMmass_nbins)]
    # write output to file
    output = np.array([DM_masses, ref_xsec*np.array(reach)]).T
    fout = run_params.fout_name + "_"+run_params.mineral_name+"_exclusion.txt"
    np.savetxt(fout, output, header="Mass [GeV], WIMP-nucleon cross section [cm^2]")
    if run_params.verbose:
        print("Finished.")
        print("Time: " + str(time.time() - start_t) + " s")
    del reach, output, fout


# projected discovery reach; parallelized
def get_discovery_reach(mass_ind):
    # get smallest allowed normalization from demanding that there
    # must be at least one signal event
    if np.sum(Spec_DM[mass_ind,:]) > 0:
        reach_min = 1. / np.sum(Spec_DM[mass_ind,:])
    else:
        return reach_max
    # run root finder
    fun = lambda xsec_relative: TS_func_discovery(mass_ind, xsec_relative) - TS_threshold_discovery
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


if run_params.output_discovery_sens:
    if run_params.verbose:
        print("")
        print("#----------------------------------")
        print("")
        print("Calculating the discovery reach...")
        print("Time: " + str(time.time() - start_t) + " s")
    if __name__ == '__main__' and run_params.Ncores_mp > 1:
        pool = mp.Pool(run_params.Ncores_mp)
        reach = pool.map_async(get_discovery_reach, range(run_params.DMmass_nbins)).get()
        pool.close()
        pool.join()
    else:
        reach = [get_discovery_reach(mass_ind) for mass_ind in range(run_params.DMmass_nbins)]
    # write output to file
    output = np.array([DM_masses, ref_xsec*np.array(reach)]).T
    fout = run_params.fout_name + "_"+run_params.mineral_name+"_discovery.txt"
    np.savetxt(fout, output, header="Mass [GeV], WIMP-nucleon cross section [cm^2]")
    if run_params.verbose:
        print("Finished.")
        print("Time: " + str(time.time() - start_t) + " s")
    del reach, output, fout

if run_params.verbose:
    print("")
    print("#----------------------------------")
