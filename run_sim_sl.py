import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm, Normalize, TwoSlopeNorm
plt.rc('image', interpolation='none')

# Basic imports
import os
import numpy as np
from copy import deepcopy
import time
from functools import partial
import corner
import json
from astropy.io import fits

# JAX
import jax
import jax.numpy as jnp
# probabilistic model
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

# variational inference
import optax  # optimizers
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal, AutoBNAFNormal

# Herculens
from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.psf import PSF
from herculens.Instrument.noise import Noise
from herculens.LightModel.light_model import LightModel
from herculens.MassModel.mass_model import MassModel
from herculens.PointSourceModel.point_source_model import PointSourceModel
from herculens.LensImage.lens_image import LensImage
from herculens.RegulModel.regul_model import RegularizationModel
from herculens.Inference.loss import Loss
from herculens.Inference.ProbModel.numpyro import NumpyroModel
from herculens.Inference.Optimization.jaxopt import JaxoptOptimizer
from herculens.Inference.Optimization.optax import OptaxOptimizer
from herculens.Analysis.plot import Plotter
from herculens.Util import image_util, param_util, plot_util

npix = 100  # number of pixel on a side
pix_scl = 0.08  # pixel size in arcsec
half_size = npix * pix_scl / 2
ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2  # position of the (0, 0) with respect to bottom left pixel
transform_pix2angle = pix_scl * np.eye(2)  # transformation matrix pixel <-> angle
kwargs_pixel = {'nx': npix, 'ny': npix,
                'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                'transform_pix2angle': transform_pix2angle}

# create the PixelGrid class
pixel_grid = PixelGrid(**kwargs_pixel)
xgrid, ygrid = pixel_grid.pixel_coordinates
extent = pixel_grid.extent

print(f"image size : ({npix}, {npix}) pixels")
print(f"pixel size : {pix_scl} arcsec")
print(f"x range    : {xgrid[0, 0], xgrid[0, -1]} arcsec")
print(f"y range    : {ygrid[0, 0], ygrid[-1, 0]} arcsec")

def get_fits_data(file_path):
    with fits.open(file_path) as hdul:
        return hdul[0].data.astype(np.float64)

psf = PSF(psf_type='GAUSSIAN', fwhm=0.3, pixel_size=pixel_grid.pixel_width)
psf_data = np.asarray(psf.kernel_point_source, dtype=np.float64)

noise = Noise(npix, npix, background_rms=1e-2, exposure_time=1000.)

# Lens mass
lens_mass_model_input = MassModel(['SIE', 'SHEAR'])

# position of the lens
cx0, cy0 = 0., 0.
# position angle, here in degree
phi = 8.0
# axis ratio, b/a
q = 0.75
# conversion to ellipticities
e1, e2 = param_util.phi_q2_ellipticity(phi * np.pi / 180, q)
# external shear orientation, here in degree
phi_ext = 54.0
# external shear strength
gamma_ext = 0.03
# conversion to polar coordinates
gamma1, gamma2 = param_util.shear_polar2cartesian(phi_ext * np.pi / 180, gamma_ext)
# print(e1, e2)
# print(gamma1, gamma2)
kwargs_lens_input = [
    {'theta_E': 1.5, 'e1': e1, 'e2': e2, 'center_x': cx0, 'center_y': cy0},  # SIE
    {'gamma1': gamma1, 'gamma2': gamma2, 'ra_0': 0.0, 'dec_0': 0.0}  # external shear
]

print(kwargs_lens_input)

# Source light
source_model_input = LightModel(['SERSIC_ELLIPSE', 'GAUSSIAN'])
beta_true = [0.05, 0.1]
kwargs_source_input = [
    {'amp': 10.0, 'R_sersic': 0.2, 'n_sersic': 2., 'e1': 0.05, 'e2': 0.05, 'center_x': beta_true[0], 'center_y': beta_true[1]},
    {'amp': 1., 'sigma': 0.05, 'center_x': beta_true[0] + 0.25, 'center_y': beta_true[1] + 0.2} # add a bit of complexity in the source galaxy
]

print(kwargs_source_input)

# Create a pixel grid for solving the lens equation
ps_grid_npix = 2 * npix + 1
ps_grid_pix_scl = (pix_scl * npix) / ps_grid_npix
ps_grid_half_size = ps_grid_npix * ps_grid_pix_scl / 2.
ps_grid_ra_at_xy_0 = ps_grid_dec_at_xy_0 = -ps_grid_half_size + ps_grid_pix_scl / 2.
ps_grid_transform_pix2angle = ps_grid_pix_scl * np.eye(2)
kwargs_ps_grid = {'nx': ps_grid_npix, 'ny': ps_grid_npix,
                  'ra_at_xy_0': ps_grid_ra_at_xy_0, 'dec_at_xy_0': ps_grid_dec_at_xy_0,
                  'transform_pix2angle': ps_grid_transform_pix2angle}
ps_grid = PixelGrid(**kwargs_ps_grid)

point_source_type_list = ['SOURCE_POSITION']
point_source_model_input = PointSourceModel(point_source_type_list, lens_mass_model_input, ps_grid)

kwargs_ps = {'ra': beta_true[0], 'dec': beta_true[1], 'amp': 2.}
kwargs_point_source_input = [kwargs_ps]

print(kwargs_point_source_input)

def json_serializer(obj):

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, jax.Array):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)

with open('sim_config.json', 'w') as f:
    json.dump({
        'kwargs_lens': kwargs_lens_input,
        'kwargs_source': kwargs_source_input,
        'kwargs_point_source': kwargs_point_source_input,
    }, f, indent=4, default=json_serializer)

# Solver hyperparameters
niter_solver = 5
triangle_scale_factor = 2
n_triangle_subdivisions = 3
kwargs_lens_equation_solver = {'nsolutions': 5, 
                               'niter': niter_solver, 
                               'scale_factor': triangle_scale_factor, 
                               'nsubdivisions': n_triangle_subdivisions}

# Point source position accuracy in image plane
ps_accuracy = ps_grid_pix_scl * (triangle_scale_factor / 4**n_triangle_subdivisions)**(niter_solver / 2.)
print(f"PS max error : {ps_accuracy:.4}")

# Generate a lensed image based on source and lens models
kwargs_numerics_simu = {'supersampling_factor': 5}
lens_image_simu = LensImage(pixel_grid, psf, noise_class=noise,
                         lens_mass_model_class=lens_mass_model_input,
                         source_model_class=source_model_input,
                         point_source_model_class=point_source_model_input,
                         kwargs_numerics=kwargs_numerics_simu,
                         kwargs_lens_equation_solver=kwargs_lens_equation_solver)

kwargs_all_input = dict(kwargs_lens=kwargs_lens_input,
                        kwargs_source=kwargs_source_input,
                        kwargs_point_source=kwargs_point_source_input)

# clean image (no noise)
image = lens_image_simu.model(**kwargs_all_input)
# noise standard deviation map estimated from the clean model
noise_map = np.sqrt(np.asarray(lens_image_simu.Noise.C_D_model(image)))

# simulated observation including noise
data = lens_image_simu.simulation(**kwargs_all_input, compute_true_noise_map=True, prng_key=jax.random.PRNGKey(42))
data_np = np.asarray(data)
image_np = np.asarray(image)

# Plotting engine
plotter = Plotter(flux_vmin=8e-3, flux_vmax=6e-1)

# inform the plotter of the data and, if any, the true source 
plotter.set_data(data)

source_input = lens_image_simu.source_surface_brightness(kwargs_source_input, de_lensed=True, unconvolved=True)
plotter.set_ref_source(source_input)

# visualize simulated products
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
img1 = ax1.imshow(image_np, origin='lower', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
plot_util.nice_colorbar(img1)
ax1.set_title("Clean lensing image")
img2 = ax2.imshow(data_np, origin='lower', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
ax2.set_title("Noisy observation data")
plot_util.nice_colorbar(img2)
img3 = ax3.imshow(data_np, origin='lower', cmap=plotter.cmap_flux)
ax3.set_title("Data (linear scale)")
plot_util.nice_colorbar(img3)
fig.tight_layout()
plt.savefig('sim_sl.png')

# plot data, noise, and psf in log scale
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
img1 = ax1.imshow(np.log10(np.clip(data_np, 1e-12, None)), origin='lower', cmap='magma')
ax1.set_title("Data (log scale)")
plot_util.nice_colorbar(img1)
img2 = ax2.imshow(np.log10(np.clip(noise_map, 1e-12, None)), origin='lower', cmap='magma')
ax2.set_title("Noise (log scale)")
plot_util.nice_colorbar(img2)
img3 = ax3.imshow(np.log10(np.clip(psf_data, 1e-12, None)), origin='lower', cmap='magma')
ax3.set_title("PSF (log scale)")
plot_util.nice_colorbar(img3)
fig.tight_layout()
plt.savefig('all_data.png')

# save data
fits.writeto('sim_sl.fits', data_np, overwrite=True)
fits.writeto('sim_sl_noise.fits', noise_map, overwrite=True)
fits.writeto('sim_sl_psf.fits', psf_data, overwrite=True)