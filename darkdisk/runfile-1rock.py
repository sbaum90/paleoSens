# 1 rock of age 100 Myr (with same total exposure as fiducial case)
# ------------------------------------------------
# Output file name
# ------------------------------------------------
Output = "result-1rock.txt"

# ------------------------------------------------
# Parameter block for sample and read-out info
# ------------------------------------------------
Mineral = "Gypsum" # Name of mineral
N_samples = 1 # Number of rock samples
Youngest_sample_age = 100 # Age of the youngest rock sample in Myr
Sample_age_spacing = 20 # Linear age spacing between different rock samples in Myr
U_concentration = 1e-11 # Uranium concentration in g/g. 1e-11 corresponds to 0.01 ppb
Sample_mass = 3e-5 # Mass of each rock sample in kg
Resolution = 10 # Resolution in angstrom

# ------------------------------------------------
# Parameters for the run setup
# ------------------------------------------------
Optimize_mass = True # Treat mass as nuisance parameter (if False, mass is fixed to fiducial value)
Cores = 4 # Number of cores used in parallelization
iminuit = True # Use iminuit for minimizer
Verbose = True # Print messages
N_track_bins = 100 # Number of track length bins
N_halo_WIMP_ms = 10000 # Number of halo wimp masses used in the iterpolator. Needs to be a large number to ensure a smooth interpolation
N_disk_WIMP_ms = 50 # Number of masses over which to compute the reach
H_tracks = False # Include hydrogen tracks
Replace_disk_spectrum_with_halo = False # Used in tests. Replaces the disk spectra with halo spectra but maintains time dependence of disk
Gaussian_likelihood = False # Replaces Poisson likelihood with Gaussian
Relative_background_systematics = 0.0 # Ignored if Gaussian likelihood is set to False

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

Atmos_neutrino_flux_unc = 1 # Relative backgrounds i.e. 1 = 100%. If None, then no constraint will be applied
DSNB_flux_unc = 1
GSNB_flux_unc = 1
Solar_neutrino_unc = 1
SampleAge_unc = 0.05
SampleMass_unc = 1e-3
U_concentration_unc = 0.1
