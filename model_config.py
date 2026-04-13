import numpy as np
import pandas as pd
from herculens_nnls.utils import convert_to_array

def lens_mass_config(image_size=None, pixel_scale=None, args=None):

    lens_mass_type_list = ['SIE', 'SHEAR']
    # lens_mass_type_list = ['GAUSSIAN'] * 5 + ['SHEAR']
    if lens_mass_type_list[0] == 'SIE':
        lens_mass_params_list = [
            {
                'theta_E': [0.4, 0.1, 0.3, 0.5],
                # 'gamma': [2.0, 0.1, 1.5, 2.5],
                'e1': [0.0, 0.1, -0.5, 0.5],
                'e2': [0.0, 0.1, -0.5, 0.5],
                'center_x': 0.0,
                'center_y': 0.0,
            },
            {
                'ra_0': 0.0,
                'dec_0': 0.0,
                'gamma1': [0.0, 0.1, -0.2, 0.2],
                'gamma2': [0.0, 0.1, -0.2, 0.2],
            }
        ]
    elif lens_mass_type_list[0] == 'EPL':
        lens_mass_params_list = [
            {
                'theta_E': [0.4, 0.1, 0.3, 0.5],
                'gamma': [2.0, 0.1, 1.5, 2.5],
                'e1': [0.0, 0.1, -0.5, 0.5],
                'e2': [0.0, 0.1, -0.5, 0.5],
                'center_x': 0.0,
                'center_y': 0.0,
            },
            {
                'ra_0': 0.0,
                'dec_0': 0.0,
                'gamma1': [0.0, 0.1, -0.2, 0.2],
                'gamma2': [0.0, 0.1, -0.2, 0.2],
            }
        ]
    elif lens_mass_type_list[0] == 'GAUSSIAN':
        lens_mass_params_list = []
        initial_list = {
            'amp': [1.0, 0.1],
            'sigma_x': [0.2, 0.1, 0, 0.5],
            'sigma_y': [0.2, 0.1, 0, 0.5],
            'center_x': 0.0,
            'center_y': 0.0,
        }
        lens_mass_params_list.append(initial_list)
        for i in range(1, 5):
            lens_mass_params_list.append(
                {
                    'amp': [1.0, 0.1],
                    'sigma_x': ['correlated', 'lens_mass', 0, 'sigma_x'],
                    'sigma_y': ['correlated', 'lens_mass', 0, 'sigma_y'],
                    'center_x': ['correlated', 'lens_mass', 0, 'center_x'],
                    'center_y': ['correlated', 'lens_mass', 0, 'center_y'],
                }
            )
        lens_mass_params_list.append(
            {
                'ra_0': 0.0,
                'dec_0': 0.0,
                'gamma1': [0.0, 0.1, -0.2, 0.2],
                'gamma2': [0.0, 0.1, -0.2, 0.2],
            }
        )

    return lens_mass_type_list, lens_mass_params_list

def lens_light_config(image_size=None, pixel_scale=None, args=None):

    num_gaussian_sets = 3
    num_gaussian_per_set = 30
    num_extra_gaussian = 0

    num_total_gaussian = num_gaussian_sets * num_gaussian_per_set + num_extra_gaussian

    lens_light_type_list = ['GAUSSIAN_ELLIPSE'] * num_total_gaussian
    
    if lens_light_type_list[0] == 'GAUSSIAN_ELLIPSE':

        print(f'Number of Gaussian sets: {num_gaussian_sets}')
        print(f'Number of Gaussian per set: {num_gaussian_per_set}')
        print(f'Number of extra Gaussian: {num_extra_gaussian}')
        print(f'Number of total Gaussian: {num_total_gaussian}')

        if pixel_scale is None:
            raise ValueError('Pixel scale is required')
        
        if image_size is None:
            raise ValueError('Image size is required')

        max_sigma = image_size * pixel_scale / 2.0
        # max_sigma = 0.5
        min_sigma = pixel_scale / 5.0

        sigma_list = 10**(np.linspace(np.log10(min_sigma), np.log10(max_sigma), num_gaussian_per_set))

        lens_light_params_list = []
        for i in range(num_gaussian_sets):
            for j in range(num_gaussian_per_set):
                geometry_head = i * num_gaussian_per_set
                if j == 0:
                    lens_light_params_list.append(
                        {
                            'amp': [1.0, 0.1],
                            'sigma': sigma_list[j], 
                            'center_x': [0.0, 0.1, -0.3, 0.3],
                            'center_y': [0.0, 0.1, -0.3, 0.3],
                            'e1': [0.0, 0.1, -0.6, 0.6],
                            'e2': [0.0, 0.1, -0.6, 0.6],
                        }
                    )
                else:
                    lens_light_params_list.append(
                        {
                            'amp': [1.0, 0.1],
                            'sigma': sigma_list[j],
                            'center_x': ['correlated', 'lens_light', geometry_head, 'center_x'],
                            'center_y': ['correlated', 'lens_light', geometry_head, 'center_y'],
                            'e1': ['correlated', 'lens_light', geometry_head, 'e1'],
                            'e2': ['correlated', 'lens_light', geometry_head, 'e2'],
                        }
                    )
    
    elif lens_light_type_list[0] == 'SERSIC_ELLIPSE':
    
        lens_light_params_list = [
            {
                'amp': [3.5, 0.1],
                'e1': [0.0, 0.1, -0.4, 0.4],
                'e2': [0.0, 0.1, -0.4, 0.4],
                'R_sersic': [0.5, 0.2, 0.01, 1.0],
                'n_sersic': [1.5, 0.5, 0.1, 8.0],
                'center_x': [0.0, 0.1, -0.5, 0.5],
                'center_y': [0.0, 0.1, -0.5, 0.5],
            },
            {
                'amp': [3.5, 0.1],
                'e1': [0.0, 0.1, -0.4, 0.4],
                'e2': [0.0, 0.1, -0.4, 0.4],
                'R_sersic': [0.5, 0.2, 0.01, 1.0],
                'n_sersic': [1.5, 0.5, 0.1, 8.0],
                'center_x': [0.0, 0.1, -0.5, 0.5],
                'center_y': [0.0, 0.1, -0.5, 0.5],
            },
        ]

    return lens_light_type_list, lens_light_params_list

def source_light_config(image_size=None, pixel_scale=None, args=None):

    num_source_light = 2
    source_light_type_list = ['SERSIC_ELLIPSE'] * num_source_light
    # source_light_type_list = ['PIXELATED']
    # source_light_type_list = ['GAUSSIAN_ELLIPSE'] * num_source_light * num_mges
    if source_light_type_list[0] == 'SERSIC_ELLIPSE':
        source_light_params_list = [
                {
                    'amp': [3.5, 0.1],
                    'e1': [0.0, 0.1, -0.3, 0.3],
                    'e2': [0.0, 0.1, -0.3, 0.3],
                    'R_sersic': [0.5, 0.1, 0.01, 1.5],
                    'n_sersic': [1.5, 0.5, 0.1, 8.0],
                    'center_x': [0.0, 0.1, -1.0, 1.0],
                    'center_y': [0.0, 0.1, -1.0, 1.0],
                },
                {
                    'amp': [3.5, 0.1],
                    'e1': [0.0, 0.1, -0.3, 0.3],
                    'e2': [0.0, 0.1, -0.3, 0.3],
                    'R_sersic': [0.5, 0.1, 0.01, 1.5],
                    'n_sersic': [1.5, 0.5, 0.1, 8.0],
                    'center_x': [0.0, 0.1, -1.0, 1.0],
                    'center_y': [0.0, 0.1, -1.0, 1.0],
                },
            ]

    elif source_light_type_list[0] == 'PIXELATED':
        source_light_params_list = []
        kwargs_pixelated_source = {
            'pixel_scale_factor': 0.3333, 
            'grid_center': (0.0, 0.0),
            'grid_shape': (2.0, 2.0), # arcsec
        }
        source_light_params_list.append(
            kwargs_pixelated_source,
        )

    return source_light_type_list, source_light_params_list

def point_source_config(image_size=None, pixel_scale=None, args=None):

    point_source_type_list = []
    point_source_params_list = []
    # point_source_type_list = ['SOURCE_POSITIONS'] * 3
    point_source_type_list = ['IMAGE_POSITIONS'] * args.num_sources
    if point_source_type_list[0] == 'SOURCE_POSITION':
        point_source_params_list += [
            # {
            #     'ra': ['correlated', 'source', 0, 'center_x'],
            #     'dec': ['correlated', 'source', 0, 'center_y'],
            #     'amp': [2.0, 0.1],
            # },
            # {
            #     'ra': ['correlated', 'source', 1, 'center_x'],
            #     'dec': ['correlated', 'source', 1, 'center_y'],
            #     'amp': [2.0, 0.1],
            # },
            {
                'ra': [0.0, 0.03, -0.1, 0.1],
                'dec': [0.0, 0.03, -0.1, 0.1],
                'amp': [2.0, 0.1],
            }, 
            {
                'ra': [0.0, 0.03, -0.1, 0.1],
                'dec': [0.0, 0.03, -0.1, 0.1],
                'amp': [2.0, 0.1],
            },
            {
                'ra': [0.0, 0.03, -0.1, 0.1],
                'dec': [0.0, 0.03, -0.1, 0.1],
                'amp': [2.0, 0.1],
            },
        ]
    elif point_source_type_list[0] == 'IMAGE_POSITIONS':
        
        ps_sources = pd.read_csv(args.image_positions_catalog)

        ras = ps_sources['ra_arcsec'].values
        decs = ps_sources['dec_arcsec'].values

        for k in range(1, args.num_sources + 1):
            raw = getattr(args, f'images_indices_{k}', None)
            if raw is None:
                raise ValueError(
                    f'Point source {k}/{args.num_sources}: missing --images_indices_{k} '
                )
            images_idx = convert_to_array(raw)
            print(f'Point source {k} catalog row indices: {images_idx}')

            images_ras = ras[images_idx]
            images_decs = decs[images_idx]
            images_num = len(images_idx)

            point_source_params_list += [
                {
                    'n_images': images_num,
                    'sigma_image': 3e-3,
                    'sigma_source': 1e-3,
                    'pos_bound': 0.1,
                    'ra': images_ras,
                    'dec': images_decs,
                    'amp': [1.0, 0.1],
                },
            ]

    return point_source_type_list, point_source_params_list