# Uses Gaussian likelihood with bin-to-bin systematic uncertainty of 0.1
# ------------------------------------------------
# Output file name
# ------------------------------------------------
Output = "result-gaussian.txt"

# ------------------------------------------------
# parameter block for sample and read-out info
# ------------------------------------------------
Mineral = "Gypsum" # Name of mineral
N_samples = 5  # Number of rock samples
Youngest_sample_age = 200  # Age of the youngest rock sample in Myr
Sample_age_spacing = 200  # Linear age spacing between different rock samples in Myr (also used as integration time for subhalo spectrum)
U_concentration = 1e-11  # Uranium concentration in g/g. 1e-11 corresponds to 0.01 ppb
Sample_mass = 0.1  # Mass of each rock sample in kg
Resolution = 150  # Resolution in angstrom

# ------------------------------------------------
# Parameters for the run setup
# ------------------------------------------------
Optimize_mass = True # Treat mass as nuisance parameter (if False, mass is fixed to fiducial value)
Cores = 20  # Number of cores used in parallelization
iminuit = True  # Use iminuit for minimizer
Verbose = True  # Print messages
N_track_bins = 100  # Number of track length bins
H_tracks = False  # Include hydrogren tracks
Replace_subhalo_spectrum_with_halo = False  # Used in tests. Replaces the subhalo spectra with halo spectra but maintains timing information of disk
Gaussian_likelihood = True  # Replaces Poisson likelihood with Gaussian
Relative_background_systematics = 0.1  # Ignored if Gaussian likelihood is set to False

Lbound_impact_parameter = 1e-4  # Lower bound on impact parameter interpolator
Ubound_impact_parameter = 1e4  # Upper bound on impact parameter interpolator
N_impact_parameter_bins = 100  # Number of impact parameters used in the interpolater. Needs to be sufficiently large to ensure smooth interpolation
N_WIMP_ms = 1000  # Number of halo wimp masses used in the interpolator. Needs to be a large number to ensure a smooth interpolation

Lbound_indepV = 10 # Lower bound on relative velocity
Ubound_indepV = 1000 # Upper bound on relative velocity
bins_indepV = 200 # Number of relative velocities used for indepedent variable

# ------------------------------------------------
# Other physical parameters
# ------------------------------------------------
WIMP_xsec = 5e-46  # spin-independent cross section in cm^-2
WIMP_m = 500  # WIMP mass in GeV
Subhalo_m = 1e6  # Subhalo mass in Msun
Concen_param = 65  # concentration parameter
T_closest_approach = 500  # Time since closest approach in Myr

# ------------------------------------------------
# Backgrounds and external constraints

# For each parameter, there is a boolean switch
# to include/not include the background,
# as well as a parameter specifying the relative
# uncertainty on the respective parameter.

# Note that background removal is only intended
# for use in tests
# ------------------------------------------------
Atmos_neutrino_background = True
DSNB_background = True
GSNB_background = True
Solar_neutrino_background = True
Neutron_background = True
SingleAlpha_background = True

Atmos_neutrino_flux_unc = 1  # Relative backgrounds i.e. 1 = 100%. If None, then no constraint will be applied
DSNB_flux_unc = 1
GSNB_flux_unc = 1
Solar_neutrino_unc = 1
SampleAge_unc = 0.05
SampleMass_unc = 1e-3
U_concentration_unc = 0.1
