import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import json
import argparse
import pandas as pd
from tqdm import tqdm

from open_clip import create_model_and_transforms, get_input_dtype
from open_clip.factory import _MODEL_CONFIGS
from open_clip_train.file_utils import pt_load
from open_clip_train.precision import get_autocast

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from hlip import visual_encoder


def get_args_parser():
    parser = argparse.ArgumentParser('Extract Embeddings', add_help=False)
    parser.add_argument('--model', default='vit_base_multiscan_h2_token1176', type=str)
    parser.add_argument('--resume', default='/pretrained/vit_base_brainmri_h2_token1176.pt', type=str)
    
    parser.add_argument('--data-root', default='/path/to/pub_brain_5')
    parser.add_argument('--input-filename', default='../../data/pub_brain_5/pub_brain_5.csv')
    parser.add_argument('--num-slices', default=144, type=int)
    parser.add_argument('--embed-root', default='/path/to/pub_brain_5_embed')
    
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--workers', default=16, type=int)
    return parser


def get_data(args, preprocess_fn):
    class PubBrain5Dataset(Dataset):
        def __init__(
            self,
            data_root,
            input_filename,
            num_slices, transform=None,
        ):
            self.data_root = data_root

            self.studies = []
            df = pd.read_csv(input_filename)
            for _, row in df.iterrows():
                if len(os.listdir(os.path.join(self.data_root, row['study']))):
                    self.studies.append(row['study'])

            self.num_slices = num_slices
            self.transform = transform
        
        def __len__(self):
            return len(self.studies)

        def __getitem__(self, idx):
            study = self.studies[idx]

            # load in imgs
            imgs = []
            for scan in [os.path.join(self.data_root, study, p) for p in os.listdir(os.path.join(self.data_root, study))]:
                img = torch.load(scan, weights_only=True)
                # check
                if len(img.shape) == 4:
                    img = img[:, :, :, 0]

                img = img[None, ...].float() / 255.0

                # process
                if self.transform:
                    img = self.transform(img)
                    img = torch.as_tensor(img).float()
                else:
                    _, _, h, w = img.shape

                    # padding to the longest side.                
                    size = max(h, w)
                    pad_h = size - h; pad_w = size - w
                    left = pad_w // 2; right = pad_w - left; top = pad_h // 2; bottom = pad_h - top
                    img = torch.nn.functional.pad(img, (left, right, top, bottom), mode="constant", value=0)

                    # resize to 256, crop to 224
                    img = torch.nn.functional.interpolate(img, size=(256, 256), mode='bilinear')
                    img = torch.nn.functional.interpolate(img[None, ...], size=(self.num_slices, 256, 256), mode='nearest-exact').squeeze(0)
                    img = img[:, :, 16:240, 16:240]

                # normalize
                normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())
                img = normalizer(img)
                imgs.append(img) 
            return study, torch.stack(imgs, dim=0)
    
        
    dataset = PubBrain5Dataset(args.data_root, args.input_filename, args.num_slices, preprocess_fn)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader


def embed(model, dataloader, args):
    model.eval()
    device = torch.device(args.device)
    autocast = get_autocast('amp', device_type=device.type)
    input_dtype = get_input_dtype('amp')
    
    with torch.inference_mode():
        for batch in tqdm(dataloader, total=len(dataloader)):
            study, image = batch
            study = study[0]
            image = image.to(device=device, dtype=input_dtype)
            with autocast():
                output = model(image=image)
                image_features = output['image_features'].detach().cpu()
                logit_scale = output['logit_scale'].detach().cpu()

            embed_dir = os.path.join(args.embed_root, args.model, str(args.num_slices), study)
            os.makedirs(embed_dir, exist_ok=True)
            torch.save(image_features, os.path.join(embed_dir, 'image_features.pt'))
            torch.save(logit_scale, os.path.join(embed_dir, 'logit_scale.pt'))
    return


def main(args):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # create model
    for _c in os.listdir('../hlip/model_configs/'):
        _m, _e = os.path.splitext(_c)
        if _e.lower() == '.json':
            with open(os.path.join('../hlip/model_configs/', _c), 'r') as f:
                model_cfg = json.load(f)
            _MODEL_CONFIGS[_m] = model_cfg
    model, _, _ = create_model_and_transforms(args.model, device=args.device, precision='amp', output_dict=True)
    checkpoint = pt_load(args.resume, map_location='cpu')
    sd = checkpoint['state_dict']
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)

    # extract embeddings
    dataloader = get_data(args, None)
    embed(model, dataloader, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Embeddings', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)