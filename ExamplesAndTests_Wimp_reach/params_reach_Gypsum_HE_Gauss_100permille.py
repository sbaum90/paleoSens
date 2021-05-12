# ------------------------------------------------
# output file name
# ------------------------------------------------
fout_name = "ExamplesAndTests_Wimp_reach/reach_HE_Gauss_100permille"


# ------------------------------------------------
# parameter block for sample and read-out info
# ------------------------------------------------
sample_age_Myr = 1e3  # age of the target sample in [Myr]
sample_mass_kg = 0.1  # mass of the target sample in [kg]
readout_resolution_Aa = 150.0  # track length resolution in [Ångström]
C238 = 1e-11  # uranium-238 concentration per weight in [g/g]
mineral_name = "Gypsum"  # name of the target mineral.
keep_H_tracks = False  # boolean variable. If True/False, tracks from hydrogen are in-/excluded in the track length spectra


# ------------------------------------------------
# external constraints on background parameters
# and sample properties.
# For each parameter, there is a boolean switch
# to include/not include the external constraint
# as well as a parameter specifying the relative
# uncertainty on the respective parameter
# ------------------------------------------------
# target sample age
ext_sample_age_bool = True
ext_sample_age_unc = 0.05
# target sample mass
ext_sample_mass_bool = True
ext_sample_mass_unc = 1e-5
# solar neutrinos
ext_nu_solar_bool = True
ext_nu_solar_unc = 1.0
# Galactic Supernova Neutrino Background
ext_nu_GSNB_bool = True
ext_nu_GSNB_unc = 1.0
# Diffuse Supernova Neutrino Background
ext_nu_DSNB_bool = True
ext_nu_DSNB_unc = 1.0
# atmospheric neutrinos
ext_nu_atm_bool = True
ext_nu_atm_unc = 1.0
# uranium-238 concentration
ext_C238_bool = True
ext_C238_unc = 0.01


# ------------------------------------------------
# parameters for the run setup
# ------------------------------------------------
TR_xmin_Aa = -1 # lower edge of smallest track length bin in [Aa]. For xmin=-1, the code uses readout_resolution_Aa/2
TR_xmax_Aa = 1e4  # upper edge of the largest track length bin in [Aa]. Should not be chosen larger than 10,000
TR_logbins = True  # set True/False for log-spaced/linear spaced track length bins
TR_nbins = 100  # number of bins. If TR_logbins == False, TR_nbins can be set to -1 in which case the bin-width is set to readout_resolution_Aa

DMmass_min_GeV = 5e-1  # smallest DM mass in [GeV] for which the limit is computed
DMmass_max_GeV = 5e3  # largest DM mass in [GeV] for which the limit is computed
DMmass_nbins = 401  # number of (log-spaced) bins for which the reach is computed

output_exclusion_sens = True  # if True, the code computes the 90% CL exclusion limit
output_discovery_sens = True  # if True, the code computes the 5-\sigma discovery sensitivity

Gaussian_likelihood = True # optional boolean parameter, use Gaussian likelihood if True
rel_bkg_sys = 0.1 # optional parameter to include systematic error on the number of background events / bin

Ncores_mp = 4  # number of cores to use for parallelized part of computation
verbose = True  # if True, code will print messages in std.out

# ------------------------------------------------
# boolean switches allowing to turn off background
# components
# This block should be used for testing only
# ------------------------------------------------
include_bkg_nu_solar = True  # solar neutrinos
include_bkg_nu_GSNB = True  # Galactic Supernova Neutrino Background
include_bkg_nu_DSNB = True # Diffuse Supernova Neutrino Background
include_bkg_nu_atm = True  # atmospheric neutrinos
include_bkg_rad_1a = True  # radiogenic single-alpha background
include_bkg_rad_neutrons = True  # radiogenic neutron background
