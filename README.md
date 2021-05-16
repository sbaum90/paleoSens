# paleoSens
Tools to project the sensitivity of paleo detectors.

This code requires [paleoSpec](https://github.com/sbaum90/paleoSpec) for the generation of the background and signal spectra. You can get it from
https://github.com/sbaum90/paleoSpec

If you use **paleoSens** for your work, please cite [arXiv:1806.05991](https://arxiv.org/abs/1806.05991), [arXiv:1811.06844](https://arxiv.org/abs/1811.06844), and [arXiv:1906.05800](https://arxiv.org/abs/1906.05800).

# WIMP_reach
This module can compute the projected (90% confidence level) exclusion limits and the (5 sigma) dissovery limit of a paleo-detector to the usual WIMP signal from the ambient dark matter halo. 

Usage: Call as `python3 WIMP_reach.py RUNFILE`, where `RUNFILE` is the name (without the .py file extension!) of the runfile. See `WIMP_default_runfile.py` for an example of the syntax and the parameters which should be entered.

Jupyter Notebooks demonstrating how to use the package and many `RUNFILE` examples be found in the "ExamplesAndTests_Wimp_reach" folder.

Per default, WIMP_reach uses a Poisson-likelihood to calculate the sensitivity. It allows to incorporate external (Gaussian) constraints on the nuissance parameters via the `ext_*` parameters in the `RUNFILE`. If the optional parameter "Gaussian_likelihood" is set to `Gaussian_likelihood = True` in the `RUNFILE`, the sensitivity is instead calculated by assuming a normal distribution of the number of events in each bin. Per default, the code will compute the variance in each bin from the Poisson error only. However, an additional parameters `rel_bkg_sys` can be included. If this parameter is set to a value different from `0`, the code will project the sensitivity by including an additional RELATIVE systematic error of the number of background events in each bin. Hence, in this case the variance of the number of events in the i-th bin is set by *var_i = N_i + (rel_bkg_sys * N_i^bkg)^2* where *N_i* is the number of events in the i-th bin, and *N_i^bkg* the number of background events in the i-th bin. Note that for the exclusion limit, *N_i = N_i^bkg*, while for the discovery reach, they differ by the contribution of the signal to the Asimov data. Finally, note that if `Gaussian_likelihood != True` the code uses a Poisson likelihood and the `rel_bkg_sys` parameter is ignored.




