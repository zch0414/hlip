"""
BraTS is a standard dataset in which every scan has a 
fixed spacing of (1 mm, 1 mm, 1 mm) and a fixed shape of (155, 240, 240). 
Therefore, no further statistical analysis is necessary for this dataset.
"""

import os
import logging
import argparse
import numpy as np
import SimpleITK as sitk
import multiprocessing as mp

import torch


def get_args_parser():
    parser = argparse.ArgumentParser('BraTS23', add_help=False)
    parser.add_argument('--num-cpus', default=1, type=int)
    parser.add_argument('--dataset', default='BraTS-GLI', type=str)
    parser.add_argument('--uint8', default=False, action='store_true')
    parser.add_argument('--root-path', default='/path/to/brats23', type=str)
    parser.add_argument('--save-path', default='/path/to/pub_brain_5/brats23/', type=str)
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


def load_nifti_file(path):
    img_sitk = sitk.ReadImage(path)
    img_sitk = reorient(img_sitk, tgt='RPI')
    img_arr = sitk.GetArrayFromImage(img_sitk) # x, y, z -> z, y, x (d, h, w)
    img_arr = clip_by_percentile(img_arr, 0.5, 99.5)
    return img_arr


def single_worker(patient_ids, set_dir, save_dir):
    for patient_id in patient_ids:
        patient_dir = os.path.join(set_dir, patient_id)
        
        patient_save_dir = os.path.join(save_dir, patient_id)
        os.makedirs(patient_save_dir, exist_ok=True)

        for series in sorted([p for p in os.listdir(patient_dir) if os.path.isfile(os.path.join(patient_dir, p)) and p.endswith('.nii.gz') and not p.startswith('.') and 'seg' not in p.lower() and 'mask' not in p.lower()]):
            series_path = os.path.join(patient_dir, series)
            series_save_path = os.path.join(patient_save_dir, series.split('.')[0] + '.pt')

            img = load_nifti_file(series_path)
            
            if args.uint8:
                img = torch.from_numpy((img * 255)).to(torch.uint8)
            else:
                img = torch.from_numpy(img)
            
            torch.save(img, series_save_path)

        logging.info(f'study-{patient_id}')


def main(args):
    for dir in ['train', 'test']:
        os.makedirs(os.path.join(args.save_path, dir), exist_ok=True)

    if args.dataset == 'BraTS-GLI':
        tumor_type = 'adult_glioma'
    elif args.dataset == 'BraTS-MEN':
        tumor_type = 'adult_meningioma'
    elif args.dataset == 'BraTS-MET':
        tumor_type = 'adult_metastasis'
    elif args.dataset == 'BraTS-PED':
        tumor_type = 'pediatric_glioma'

    data_dir = os.path.join(args.root_path, args.dataset)
    for dir in sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]):
        if 'Train' in dir:
            save_dir = os.path.join(args.save_path, 'train', tumor_type)
            os.makedirs(save_dir, exist_ok=True)
        elif 'Validation' in dir:
            save_dir = os.path.join(args.save_path, 'test', tumor_type)
            os.makedirs(save_dir, exist_ok=True)
        else:
            continue
        
        set_dir = os.path.join(data_dir, dir)
        patient_ids = sorted([d for d in os.listdir(set_dir) if os.path.isdir(os.path.join(set_dir, d))])
        patient_ids_chunks = np.array_split(patient_ids, args.num_cpus)
        input_chunks = [(patient_ids_chunk, set_dir, save_dir) for patient_ids_chunk in patient_ids_chunks]
        with mp.Pool(processes=args.num_cpus) as pool:
            pool.starmap(single_worker, input_chunks)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BraTS23', parents=[get_args_parser()])
    args = parser.parse_args()

    # set logging format
    logging.basicConfig(
        filename=f'./logs/process_dataset/{args.dataset}.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f'{args}\n')

    main(args)