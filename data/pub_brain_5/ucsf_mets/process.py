import os
import logging
import argparse
import numpy as np
import SimpleITK as sitk
import multiprocessing as mp

import torch


def get_args_parser():
    parser = argparse.ArgumentParser('UCSF Mets', add_help=False)
    parser.add_argument('--num-cpus', default=1, type=int)
    parser.add_argument('--uint8', default=False, action='store_true')
    parser.add_argument('--statistic', default=False, action='store_true')
    parser.add_argument('--root-path', default='/path/to/ucsf_mets/', type=str)
    parser.add_argument('--save-path', default='/path/to/pub_brain_5/ucsf_mets/', type=str)
    return parser


def reorient(img_sitk, tgt='RPI'):
    """
    Reorientation from src -> tgt for the input img.
    Although this function is flexible enough for tgt,
    it is important to follow the standard orientation order as:
    'RPI' for Python; 'LPS' for 3D Slicer.
    Parameters:
        img: An sitk image of shape [x, y, z].
        tgt: A string of target orentations.
    Returns:
        img: An sitk image after transposing.
    """
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(tgt)
    return orienter.Execute(img_sitk)


def clip_by_percentile(img, min, max):
    """ Percentile Clip
    """
    lower = np.percentile(img, min)
    upper = np.percentile(img, max)
    img = np.clip(img, lower, upper)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def compute_spacing(img_sitk):
    """Returns the spacing of the given SimpleITK image.
    """
    spacing_sitk = img_sitk.GetSpacing()
    return np.array(spacing_sitk, dtype=float)


def transpose2dhw(img_arr, spacing_sitk):
    """ 
    Transpose img as:
    - D: Through-plane (with the lowest resolution / largest spacing)
    - H and W: In-plane (with higher resolution)
    Parameters:
        img: An axial-like numpy array of shape [z, y, x].
        spacing: A sitk spacing for [x, y, z].
    Return:
        img: A numpy array after transposing axes.
        view: A string indicate different views.
    """
    spacing_array = spacing_sitk[::-1] # x,y,z -> z,y,x

    # Identify the dimension with the largest spacing -> that axis becomes D (through-plane)
    _max_spacing_side = np.argmax(spacing_array)
    if _max_spacing_side == 0:  
        # z-axis has largest spacing -> axial
        # (keep img_arr unchanged)
        view = 'axial'
    elif _max_spacing_side == 1:  
        # y-axis has largest spacing -> coronal
        img_arr = np.transpose(img_arr, (1, 0, 2))  # (y, z, x)
        view = 'coronal'
    else:  
        # x-axis has largest spacing -> sagittal
        img_arr = np.transpose(img_arr, (2, 0, 1))  # (x, z, y)
        view = 'sagittal'
    return img_arr, view


def load_nifti_file(path):
    img_sitk = sitk.ReadImage(path)
    img_sitk = reorient(img_sitk, tgt='RPI')
    spacing_sitk = compute_spacing(img_sitk)
    img_arr = sitk.GetArrayFromImage(img_sitk) # x, y, z -> z, y, x (d, h, w)
    img_arr, view = transpose2dhw(img_arr, spacing_sitk)
    img_arr = clip_by_percentile(img_arr, 0.5, 99.5)
    return img_arr, view


def single_worker(patient_ids, root_dir, save_dir):
    for patient_id in patient_ids:
        patient_dir = os.path.join(root_dir, patient_id)
        patient_save_dir = os.path.join(save_dir, patient_id)
        os.makedirs(patient_save_dir, exist_ok=True)

        for series in sorted([p for p in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, p)) and not p.startswith('.') and 'seg' not in p.lower() and 'mask' not in p.lower()]):
            series_path = os.path.join(patient_dir, series, 'nifti', 'SERIES.nii.gz')
            series_save_path = os.path.join(patient_save_dir, series + '.pt')

            img, view = load_nifti_file(series_path)
            
            if args.uint8:
                img = torch.from_numpy((img * 255)).to(torch.uint8)
            else:
                img = torch.from_numpy(img)
            
            logging.info(f"study-{patient_id}_series-{series}_view-{view}_shape-{[*img.shape]}")
            
            if args.statistic:
                continue
            else:
                torch.save(img, series_save_path)


def main(args):
    os.makedirs(os.path.join(args.save_path), exist_ok=True)

    patient_ids = np.array([p for p in os.listdir(args.root_path) if os.path.isdir(os.path.join(args.root_path, p)) and not p.startswith('.')])
    patient_ids_chunks = np.array_split(patient_ids, args.num_cpus)
    input_chunks = [(patient_ids_chunk, args.root_path, args.save_path) for patient_ids_chunk in patient_ids_chunks]
    with mp.Pool(processes=args.num_cpus) as pool:
        pool.starmap(single_worker, input_chunks)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('UCSF Mets', parents=[get_args_parser()])
    args = parser.parse_args()

    # set logging format
    logging.basicConfig(
        filename=f'./process.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f'{args}\n')

    main(args)