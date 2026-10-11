# Numerical protocol

Schnakenberg reactions f=.1−u+u²v, h=.9−u²v; diffusion Du1, Dv1/5/10. All quantities dimensionless. Uniform equilibrium (1,.9), reaction Jacobian [[.8,1],[-1.8,-1]]. Periodic square L40, grid64, dt.02, explicit reaction/implicit Fourier diffusion. Independent Gaussian zero-mean initial perturbations with SD.001, seeds0–4, reused across diffusion ratios. No clipping.

Record spatial SD every time unit, snapshots0/25/50/100/200. Final field settled if SD<1e-8 or SD range and field RMS change over last25 units are each <2% of final SD. Extend unsettled runs once to400 and retain unresolved status. Largest individual nonzero Fourier mode estimates wavelength only if SD>1e-8; do not assign wavelength to numerical noise.

Controls: exact equilibrium, diffusion-only smoothing/mean preservation, no-diffusion comparison with local ODE solver. Check dt.01 for seed0 Dv10 (same pixel noise); compare64/128 grids on same smooth analytic initial field. Compare final SD within10% and wavelength within20%; report rather than hide failures. Uniform-control absolute error tolerance1e-10. Refine at same final observation time.

Analytical dispersion curve predicts early growth, not final nonlinear geometry. Runs, not pixels, are experimental units. No external biological data.
